"""Gemini 로 설교 편집 제안 받기 + 정확한 위치로 맞추기

흐름
  1) 설교 음성을 Gemini Files API 로 올림
  2) 번호 붙은 받아쓰기 원고와 함께 "지울 만한 곳"을 JSON 으로 받음
  3) 제안마다 정확한 구간을 계산
     - 원고에 있는 말(quote) → 받아쓰기 단어 시각에서 찾아 쉼 안에서 자르는 구간
     - 원고에 없는 소리(어/음 등) → 제안 시각 근처의 '쉼으로 둘러싸인 목소리 덩어리'
제안은 자동 적용하지 않고 편집 화면에서 사람이 확정한다.
"""
import difflib, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

TYPES = {
    "filler": "추임새",
    "misspeak": "말실수",
    "restart": "다시 말함",
    "accidental_repeat": "실수 반복",
    "intentional_repeat": "의도적 반복",
    "callout": "청중 호응",
    "announcement": "광고·안내",
    "sermon_end": "설교 끝",
}
REMOVABLE = {"filler", "misspeak", "restart", "accidental_repeat", "callout", "announcement"}

SYSTEM = """당신은 한국 기독교 라디오(극동방송)의 설교 편집자입니다.
주일 예배 설교 녹음을 라디오 송출용으로 다듬기 위해, 지우면 좋은 곳을 찾아 제안합니다.
설교 내용 자체는 절대 줄이거나 요약하지 않습니다. 듣기 거슬리는 군더더기와 예배당 청중에게만 의미 있는 말만 찾습니다.

찾을 것(type):
- filler: "어", "음", "에", "그", "저" 처럼 뜻 없이 끼어든 추임새. 받아쓰기에는 빠져 있는 경우가 많으니 음성을 직접 듣고 찾으세요.
- misspeak: 잘못 말했다가 바로 고쳐 말한 경우의 '틀린 쪽'(예: "감사함은 했고, 빌립보서…"에서 "했고"). 고친 말은 남깁니다.
- restart: 문장을 시작했다가 끊고 처음부터 다시 말한 경우의 '끊긴 앞부분'.
- accidental_repeat: 말이 꼬여서 같은 단어·구절을 실수로 되풀이한 것(예: "저는 저는").
- intentional_repeat: 강조·성경 구절 다시 읽기·대구법처럼 일부러 반복한 것. 지우면 안 되므로 '지우지 말 것'으로 표시만 합니다.
- callout: 예배당 청중에게 반응을 요구하는 말("아멘이십니까?", "할렐루야", "옆 사람에게 말해 보세요", "다 같이 따라 하십시다" 등). 라디오 청취자에게는 어색합니다.
- announcement: 설교와 무관한 교회 광고·행사 안내·인사.
- sermon_end: 설교 본문의 마지막 문장(보통 "…주님의 이름으로 축복합니다"). 그 뒤의 찬양 안내·기도 안내는 송출하지 않습니다. 딱 하나만 표시하세요.

규칙:
- 원고는 [L번호 분:초] 형식의 받아쓰기입니다. 받아쓰기는 틀린 글자가 있을 수 있으니 음성을 기준으로 판단하세요.
- line_id 는 해당 줄 번호(정수)입니다.
- quote 에는 지울 말을 '원고에 적힌 글자 그대로' 적습니다. 원고에 없는 소리(추임새 등)는 quote 를 "" 로 두고, 들리는 소리를 heard 에 적습니다.
- approx_time 에는 그 소리가 시작되는 시각을 음성 기준 "분:초.소수" (예: "15:14.1") 로 적습니다.
- reason 은 한국어 25자 이내.
- confidence 는 0~1. 확실하지 않으면 낮게 주세요. 애매하면 제안하지 않는 쪽을 택하세요.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(TYPES)},
                    "line_id": {"type": "integer"},
                    "quote": {"type": "string"},
                    "heard": {"type": "string"},
                    "approx_time": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "line_id", "quote", "approx_time", "reason", "confidence"],
            },
        }
    },
    "required": ["items"],
}


# ---- 설정 -------------------------------------------------------------------
def load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for raw in open(p, encoding="utf-8-sig"):
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()
    return os.environ.get("GEMINI_API_KEY", "").strip()


def available():
    return bool(load_env())


def client():
    from google import genai
    key = load_env()
    if not key:
        raise RuntimeError("Gemini API 키가 없습니다. _도구\\.env 의 GEMINI_API_KEY= 뒤에 키를 넣어 주세요.")
    return genai.Client(api_key=key)


def pick_model(c, prefer="flash"):
    """설정(GEMINI_MODEL)이 있으면 그것, 없으면 사용 가능한 모델 중 이름에 prefer 가 들어간 최신 버전"""
    env = os.environ.get("GEMINI_MODEL", "").strip()
    if env:
        return env
    names = []
    for m in c.models.list():
        n = m.name.split("/")[-1]
        acts = getattr(m, "supported_actions", None) or []
        if "generateContent" not in acts or not n.startswith("gemini-"):
            continue
        if any(x in n for x in ("tts", "image", "live", "embedding", "audio", "lite", "robotics", "computer")):
            continue
        if prefer in n:
            names.append(n)
    if not names:
        raise RuntimeError("사용할 수 있는 Gemini 모델을 찾지 못했습니다. .env 의 GEMINI_MODEL 에 모델 이름을 적어 주세요.")

    def ver(n):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (float(m.group(1)) if m else 0, "preview" not in n and "exp" not in n, -len(n))
    return max(names, key=ver)


# ---- Gemini 호출 --------------------------------------------------------------
def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def ask_gemini(mp3, lines, model=None, log=print):
    """lines: [(start, end, text), ...]  →  (모델 이름, [제안 dict])"""
    from google.genai import types
    c = client()
    model = model or pick_model(c)
    log(f"  Gemini 모델: {model}")
    log("  설교 음성 올리는 중…")
    f = c.files.upload(file=mp3, config=types.UploadFileConfig(mime_type="audio/mpeg"))
    t0 = time.time()
    while f.state == types.FileState.PROCESSING and time.time() - t0 < 300:
        time.sleep(3)
        f = c.files.get(name=f.name)
    if f.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Gemini 파일 처리 실패: {f.state}")
    script = "\n".join(f"[L{i} {_fmt(s)}] {t}" for i, (s, e, t) in enumerate(lines))
    prompt = ("아래는 이 설교 음성의 받아쓰기 원고입니다. 음성을 처음부터 끝까지 듣고, 규칙에 따라 편집 제안을 JSON 으로 주세요.\n\n"
              + script)
    log("  Gemini 분석 중… (1~3분)")
    try:
        r = c.models.generate_content(
            model=model, contents=[f, prompt],
            config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                               response_json_schema=SCHEMA))
    finally:
        try:
            c.files.delete(name=f.name)
        except Exception:
            pass
    data = json.loads(r.text)
    usage = getattr(r, "usage_metadata", None)
    if usage:
        log(f"  사용 토큰: 입력 {usage.prompt_token_count:,} / 출력 {usage.candidates_token_count or 0:,}")
    return model, data.get("items", [])


# ---- 위치 맞추기 ---------------------------------------------------------------
def _norm(s):
    return re.sub(r"[\s.,?!~'\"·…\-]", "", s or "")


def _ptime(s):
    m = re.match(r"\s*(\d+):(\d+(?:\.\d+)?)", s or "")
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _find_quote(quote, words):
    """words: [(s, e, w)] 에서 quote 와 가장 비슷한 연속 단어 범위 → (i, j, 점수)"""
    q = _norm(quote)
    if not q or not words:
        return None
    nw = [_norm(w[2]) for w in words]
    best = None
    for i in range(len(words)):
        acc = ""
        for j in range(i, min(len(words), i + 25)):
            acc += nw[j]
            if len(acc) > len(q) * 1.6 + 2:
                break
            sc = difflib.SequenceMatcher(None, q, acc).ratio()
            if best is None or sc > best[2]:
                best = (i, j, sc)
    return best if best and best[2] >= 0.6 else None


def _voice_island(au, t, words, span=3.0):
    """t 근처의 '쉼으로 둘러싸인 짧은 소리 덩어리' 중 받아쓰기 단어가 덜 덮는 것"""
    sil = au.silences(t - span, t + span, 0.06)
    best = None
    ws = np.array([(w[0], w[1]) for w in words]) if words else np.zeros((0, 2))
    for (s0, e0), (s1, e1) in zip(sil[:-1], sil[1:]):
        a, b = e0, s1
        if not (0.06 <= b - a <= 1.2):
            continue
        cover = np.clip(np.minimum(ws[:, 1], b) - np.maximum(ws[:, 0], a), 0, None).sum() / (b - a) if len(ws) else 0
        score = abs((a + b) / 2 - t) + cover * 2.0
        if best is None or score < best[0]:
            best = (score, a, b, cover)
    return best


def align(items, lines, au, f):
    """Gemini 제안 → 편집 화면용 제안 목록.
    lines: 편집 화면 DATA.lines (dict: s, e, t, words=[[s,e,w]])
    f: febc_edit 모듈 (word_cut_region 사용)"""
    out = []
    for k, it in enumerate(items):
        typ = it.get("type")
        if typ not in TYPES:
            continue
        li = it.get("line_id")
        li = li if isinstance(li, int) and 0 <= li < len(lines) else None
        t_hint = _ptime(it.get("approx_time"))
        sug = {"id": k, "type": typ, "label": TYPES[typ], "line": li, "quote": it.get("quote", ""),
               "heard": it.get("heard", ""), "reason": it.get("reason", ""),
               "conf": round(float(it.get("confidence", 0)), 2), "kind": "note", "s": None, "e": None}
        if typ in ("intentional_repeat", "sermon_end"):
            if li is not None:
                sug["s"], sug["e"] = lines[li]["s"], lines[li]["e"]
            out.append(sug)
            continue
        # 1) 원고에 있는 말: 근처 줄들의 단어에서 찾기
        found = None
        if sug["quote"]:
            cand = range(max(0, (li or 0) - 1), min(len(lines), (li or 0) + 2)) if li is not None else range(len(lines))
            for ci in cand:
                ws = lines[ci]["words"]
                m = _find_quote(sug["quote"], ws)
                if m and (found is None or m[2] > found[3]):
                    found = (ci, m[0], m[1], m[2])
        if found:
            ci, i, j, sc = found
            ws = lines[ci]["words"]
            if i == 0 and j == len(ws) - 1:          # 줄 전체
                sug.update(kind="line", line=ci, s=lines[ci]["s"], e=lines[ci]["e"])
            else:
                a, b = f.word_cut_region(au, ws[i][0], ws[j][1])
                sug.update(kind="region", line=ci, s=round(a, 2), e=round(b, 2), words=[i, j])
            sug["match"] = round(sc, 2)
            out.append(sug)
            continue
        # 2) 원고에 없는 소리(추임새 등): 제안 시각 근처의 목소리 덩어리
        if t_hint is not None:
            near = [w for L in lines for w in L["words"] if abs(w[0] - t_hint) < 6]
            isl = _voice_island(au, t_hint, near)
            if isl:
                _, a, b, cover = isl
                a2, b2 = f.word_cut_region(au, a, b)
                sug.update(kind="region", s=round(a2, 2), e=round(b2, 2), match=round(1 - cover, 2))
                if sug["line"] is None:
                    sug["line"] = min(range(len(lines)), key=lambda x: abs(lines[x]["s"] - a))
                out.append(sug)
                continue
        sug["kind"] = "unplaced"   # 위치를 못 찾음: 표시만
        out.append(sug)
    out.sort(key=lambda s: (s["s"] if s["s"] is not None else 1e9))
    return out


def review(mp3, data_lines, au, f, model=None, log=print):
    """편집 화면 데이터(lines)로 Gemini 제안을 받아 정렬까지. 실패하면 예외."""
    simple = [(L["s"], L["e"], L["t"]) for L in data_lines]
    model, items = ask_gemini(mp3, simple, model, log)
    sugs = align(items, data_lines, au, f)
    n = {}
    for s in sugs:
        n[s["label"]] = n.get(s["label"], 0) + 1
    log("  Gemini 제안: " + (", ".join(f"{k} {v}" for k, v in n.items()) or "없음"))
    return model, items, sugs
