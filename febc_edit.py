"""극동방송 송출용 설교 MP3 편집기

규칙 (2026-11-15 원본 vs 극동방송 송출본 비교로 도출):
  1. 앞: 첫 말소리 1초 전부터 시작
  2. 끝: 설교 마지막 "…이름으로 축복합니다" 다음의 찬양/기도 안내 멘트와 찬양을 삭제
  3. 청중 호응 멘트("아멘이십니까?", "할렐루야")는 기본 삭제
  4. 숨소리는 자동 삭제
  5. 쉼: 0.3초 이상 쉼을 66%로 줄이고 최대 0.9초로 제한
  6. 음량: 압축 후 -25.5 LUFS, 피크 약 -7.5 dBFS
  7. MP3 192kbps / 48kHz / 스테레오, 25분 이내

사용법:
  1) 원고 만들기 : python febc_edit.py --script "원본.mp3"
                   -> 원본_원고.txt, 원본_편집.html (브라우저 편집 화면)
  2) 편집 화면에서 중복 의심(빨간색) 등을 들어보고 줄/단어를 지운 뒤 [원고 저장]
     (또는 메모장으로 원고.txt 의 줄 앞에 # 표시)
  3) 송출 파일   : python febc_edit.py "원본.mp3" 또는 "원본_원고.txt"  -> 원본_1.mp3
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, warnings
import numpy as np
import imageio_ffmpeg

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 48000
HOP = SR // 100  # 10 ms
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 튜닝 값 -------------------------------------------------------------
THR_BELOW_SPEECH = 19   # 쉼 판정: 말소리 대표 음량(상위 30%)보다 이만큼 작으면 쉼
LEAD_SEC = 1.0          # 첫 말소리 앞에 남길 여백
TAIL_SEC = 0.8          # 마지막 말 뒤에 남길 여백
JOIN_SEC = 0.35         # 문장을 잘라낸 자리 양쪽에 남길 쉼
WORD_JOIN_SEC = 0.25    # 단어를 잘라낸 자리 양쪽에 남길 쉼(최대)
FADE_OUT = 0.3
PAD_START = 2.0         # 맨 앞에 넣는 무음(초): 기존 여백 1초와 합쳐 약 3초 뒤 설교 시작
PAD_END = 2.0           # 맨 끝에 넣는 무음(초)
MIN_PAUSE = 0.30        # 이보다 짧은 쉼은 건드리지 않음
PAUSE_KEEP = 0.80       # 문장 중간 쉼: 유지 비율
PAUSE_MAX = 1.00        # 문장 중간 쉼: 최대 길이
SENT_KEEP = 0.85        # 문장 사이 쉼: 유지 비율
SENT_MIN = 0.60         # 문장 사이 쉼: 최소(원래 쉼이 더 짧으면 그대로)
SENT_MAX = 1.20         # 문장 사이 쉼: 최대
BREATH_ATTEN_DB = 12.0  # 숨소리: 잘라내지 않고 소리만 이만큼 줄임(호흡 리듬 유지, 너무 깊으면 구멍처럼 들림)
SENT_END = re.compile(r"([.?!]|(니다|니까|어요|에요|예요|해요|세요|지요|죠|군요|네요|습니다|시다|시오|요|다|까))$")
XFADE = 0.030           # 컷 지점 크로스페이드(배경 소리가 튀지 않게 부드럽게)
REF_IN_LUFS = -18.6     # 압축 전에 녹음 크기를 이 값으로 맞춤(주마다 녹음 크기가 달라도 압축 정도가 같도록)
NOISE_MODE = "denoise"  # "denoise": 전체에 일정한 잡음 제거(자연스러움, 기본) / "gate": 쉼 구간만 낮춤(끊기는 느낌)
DENOISE = "afftdn=nr=10:nf=-55:tn=1"
PAUSE_ATTEN_DB = 14.0 # 말 없는 쉼 구간을 이만큼 낮춤(실내 잡음 억제, 완전 무음은 아님)
PAUSE_GUARD = 0.08      # 말소리 앞뒤 이 시간은 낮추지 않음(말끝·첫소리 보호)
PAUSE_RAMP = 0.07       # 쉼 구간 낮추기/숨소리 줄이기가 들어가고 나오는 시간(초)
TARGET_LUFS = -25.5
LIMIT_DB = -8.0
MAX_LEN_SEC = 25 * 60   # 극동방송 송출 길이 기준: 25분 이내
COMPRESSOR = "acompressor=threshold=-30dB:ratio=4:attack=20:release=250:detection=rms:knee=4"
ASR_MODEL = "small"

REMOVE_BREATH = True    # 숨소리 자동 삭제
ISLAND_GAP = 0.12       # 앞뒤로 이만큼 이상 쉼이 있는 짧은 소리 덩어리만 숨소리 후보
BREATH_MAX_LEN = 0.8
BREATH_BELOW_SPEECH = 12   # 말소리보다 12dB 이상 작고
BREATH_FLATNESS = 0.08     # 잡음성(평탄도) 높고
BREATH_VOICE_MAX = 0.45    # 목소리 성분 낮음

DUP_WINDOW = 60         # 이 시간(초) 안의 비슷한 줄을 중복 의심
DUP_SIM = 0.6           # 글자 유사도 기준
DUP_MIN_CHARS = 6
DUP_MIN_SHARED = 6      # 겹치는 글자쌍이 이 이상이어야 (어미만 같은 줄 제외)

END_WORDS = re.compile(r"축\s*(복|원)\s*(합|드립)니다|축복\s*합니다")
MENT_WORDS = re.compile(r"(찬양|잔향|기도)\s*(하시겠|하겠|합시다|하십시|드리겠)")
CALLOUT_WORD = re.compile(r"^(할렐루야|아멘|아메)")
CALLOUT_LINE = re.compile(r"^(할렐루야|아멘|아메[니이]십니까|아멘이십니까|아멘입니까|아멘하십니까|믿습니까)+$")
SCRIPT_SUFFIX = "_원고.txt"
HTML_SUFFIX = "_편집.html"
LINE_RE = re.compile(r"^\s*(#)?\s*\[\s*(\d+):(\d+(?:\.\d+)?)\s*~\s*(\d+):(\d+(?:\.\d+)?)\s*\]\s?(.*)$")
WORD_RE = re.compile(r"^\s*-\s*\[\s*(\d+):(\d+(?:\.\d+)?)\s*~\s*(\d+):(\d+(?:\.\d+)?)\s*\]")
# -------------------------------------------------------------------------


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, check=True, **kw)


def load(path):
    raw = run([FF, "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def frame_db(mono, smooth=5):
    n = len(mono) // HOP
    f = mono[: n * HOP].reshape(n, HOP)
    d = 10 * np.log10((f.astype(np.float64) ** 2).mean(1) + 1e-12)
    if smooth > 1:  # 잡음 깜빡임 제거용 이동 중앙값
        pad = np.pad(d, smooth // 2, mode="edge")
        d = np.median(np.lib.stride_tricks.sliding_window_view(pad, smooth), axis=1)
    return d


def runs(mask):
    """True 구간 목록 [(start, end)) (프레임 단위)"""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def tc(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def tc2(sec):
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


class Audio:
    def __init__(self, path):
        self.path = path
        self.x = load(path)
        self.mono = self.x.mean(1)
        self.db = frame_db(self.mono)
        self.total = len(self.x) / SR
        self.ref = np.percentile(self.db, 70)
        self.thr = self.ref - THR_BELOW_SPEECH
        self.quiet = self.db < self.thr               # 진짜 무음
        self.drop = np.zeros(len(self.db), bool)      # 숨소리: 쉼으로 취급해 잘라냄

    def silences(self, t0, t1, min_len=0.1):
        a, b = max(0, int(t0 * 100)), min(len(self.db), int(t1 * 100))
        if b <= a:
            return []
        m = (self.quiet | self.drop)[a:b]
        return [((s + a) / 100, (e + a) / 100) for s, e in runs(m) if (e - s) / 100 >= min_len]

    def mark_drop(self, t0, t1):
        self.drop[int(t0 * 100):int(np.ceil(t1 * 100))] = True

    def pause_gain(self, s, e):
        """샘플 [s, e) 에 곱할 이득: 말 없는 쉼 구간만 PAUSE_ATTEN_DB 낮춤(부드럽게)"""
        if not hasattr(self, "_g"):
            k = int(PAUSE_GUARD * 100)
            silent = self.quiet | self.drop
            speech = np.convolve((~silent).astype(np.int16), np.ones(2 * k + 1, np.int16), "same") > 0
            att = PAUSE_ATTEN_DB if NOISE_MODE == "gate" else 0.0
            g = np.where(speech, 1.0, 10 ** (-att / 20))
            g = np.where(self.drop & ~speech, 10 ** (-BREATH_ATTEN_DB / 20), g)  # 숨소리는 더 깊게
            r = max(1, int(PAUSE_RAMP * 100) // 2)  # 램프(초) → 이동평균 반폭(프레임)
            self._g = np.convolve(np.pad(g, r, mode="edge"), np.ones(2 * r + 1) / (2 * r + 1), "valid").astype(np.float32)
        t = np.arange(s, e, dtype=np.float64) / HOP - 0.5
        return np.interp(t, np.arange(len(self._g)), self._g).astype(np.float32)


# ---- 숨소리 탐지 ----------------------------------------------------------
def island_features(au, a, b):
    """(말소리 대비 최대음량 dB, 평탄도, 목소리 성분)"""
    x = au.mono[int(a * SR):int(b * SR)].astype(np.float64)
    n, hop = 1024, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x) + 1))
    win = np.hanning(n)
    fr = np.array([x[i:i + n] * win for i in range(0, len(x) - n + 1, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(n, 1 / SR)
    Sb = S[:, (freqs > 100) & (freqs < 8000)]
    flat = float(np.median(np.exp(np.log(Sb).mean(1)) / Sb.mean(1)))
    lo, hi = int(SR / 400), int(SR / 60)
    voice = []
    for w in fr:
        w = w - w.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * n)) ** 2)[:n]
        voice.append(ac[lo:hi].max() / ac[0] if ac[0] > 0 else 0)
    peak = float(au.db[int(a * 100):max(int(a * 100) + 1, int(b * 100))].max() - au.ref)
    return peak, flat, float(np.median(voice))


def detect_breaths(au, t0, t1):
    """앞뒤가 쉼으로 둘러싸인, 작고 잡음 같고 목소리 성분 없는 소리 덩어리"""
    sil = au.silences(t0, t1, ISLAND_GAP)
    out = []
    for (s0, e0), (s1, e1) in zip(sil[:-1], sil[1:]):
        a, b = e0, s1
        if not (0.02 <= b - a <= BREATH_MAX_LEN):
            continue
        peak, flat, voice = island_features(au, a, b)
        if peak < -BREATH_BELOW_SPEECH and flat >= BREATH_FLATNESS and voice < BREATH_VOICE_MAX:
            out.append((a, b))
    return out


def find_music_start(au):
    """끝부분의 '쉼 없는 연속음(찬양)' 시작 지점(초). 없으면 None."""
    db, thr = au.db, au.thr
    blk = 200  # 2초 블록
    nb = len(db) // blk
    cont = []
    for b in range(nb):
        seg = db[b * blk:(b + 1) * blk]
        gap = max((e - s for s, e in runs(seg < thr)), default=0)
        cont.append(gap <= 5 and (seg > -40).mean() > 0.95)
    lo = max(0, nb - 240)  # 마지막 8분 안에서만
    start = None
    for b in range(nb - 2, lo - 1, -1):
        if cont[b] or (b > 0 and cont[b - 1] and b + 1 < nb and cont[b + 1]):
            start = b
        else:
            break
    if start is None or (nb - start) < 10:  # 20초 미만이면 음악 아님
        return None
    return start * blk / 100


# ---- 음성인식 -------------------------------------------------------------
def transcribe(path, t0=0.0, t1=None, vad=False):
    from faster_whisper import WhisperModel
    cmd = [FF, "-v", "error", "-ss", str(t0)] + (["-to", str(t1)] if t1 else []) + \
          ["-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    audio = np.frombuffer(run(cmd).stdout, np.float32)
    m = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(audio, language="ko", beam_size=5, word_timestamps=True, vad_filter=vad)
    out = []
    for s in segs:
        words = [(w.start + t0, w.end + t0, w.word) for w in (s.words or [])]
        out.append((s.start + t0, s.end + t0, s.text.strip(), words))
        if t1 is None:
            print(f"\r  받아쓰는 중… {tc(s.end + t0)}", end="", flush=True)
    if t1 is None:
        print()
    return out


def split_lines(segs):
    """인식 결과를 문장 단위 줄 [(start, end, text, words)]로 나눔. 호응 멘트는 따로 한 줄."""
    lines = []
    for s0, s1, text, words in segs:
        if not words:
            lines.append((s0, s1, text, [])); continue
        cur = []

        def flush():
            t = "".join(x[2] for x in cur).strip()
            if t:
                lines.append((cur[0][0], cur[-1][1], t, list(cur)))
            cur.clear()

        for i, w in enumerate(words):
            callout = bool(CALLOUT_WORD.match(w[2].strip()))
            if callout and cur and not CALLOUT_WORD.match(cur[-1][2].strip()):
                flush()
            cur.append(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            gap = 99 if nxt is None else nxt[0] - w[1]
            dur = w[1] - cur[0][0]
            nxt_callout = nxt is not None and CALLOUT_WORD.match(nxt[2].strip())
            if (nxt is None or w[2].strip().endswith((".", "?", "!")) or gap >= 0.6
                    or (dur > 12 and gap >= 0.25) or (callout and not nxt_callout)):
                flush()
    return lines


def is_callout(text):
    return bool(CALLOUT_LINE.match(re.sub(r"[\s.,?!~]", "", text)))


def sermon_end_index(lines, music_start, total):
    """설교 마지막 줄 번호와 근거"""
    limit = music_start if music_start else total
    window = limit - 180
    for i in range(len(lines) - 1, -1, -1):
        s, e, t = lines[i][:3]
        if window <= s < limit and END_WORDS.search(t):
            return i, f"'{t}'"
    for i, l in enumerate(lines):
        s, e, t = l[:3]
        if window <= s < limit and MENT_WORDS.search(t):
            return i - 1, f"안내 멘트 '{t}' 직전 (축복합니다 못 찾음 - 확인 필요)"
    last = max([i for i, l in enumerate(lines) if l[0] < limit], default=len(lines) - 1)
    return last, "끝맺음 문장을 못 찾아 음악 직전에서 자름 (확인 필요)"


# ---- 중복 의심 -------------------------------------------------------------
def _norm(t):
    return re.sub(r"[\s.,?!~'\"·…-]", "", t)


def _sim(a, b):
    A = [a[i:i + 2] for i in range(len(a) - 1)]
    B = [b[i:i + 2] for i in range(len(b) - 1)]
    if not A or not B:
        return 0.0, 0
    from collections import Counter
    ca, cb = Counter(A), Counter(B)
    shared = sum((ca & cb).values())
    return 2 * shared / (len(A) + len(B)), shared


def find_duplicates(lines):
    """줄마다 (비슷한 줄 번호 또는 -1, 종류, 반복 단어 번호들)"""
    norm = [_norm(l[2]) for l in lines]
    out = []
    for i, l in enumerate(lines):
        dup, kind = -1, ""
        if len(norm[i]) >= DUP_MIN_CHARS:
            best = 0
            for j in range(i - 1, -1, -1):
                if l[0] - lines[j][0] > DUP_WINDOW:
                    break
                s, shared = _sim(norm[i], norm[j])
                if s >= DUP_SIM and shared >= DUP_MIN_SHARED and s > best:
                    best, dup, kind = s, j, "similar"
        # 말을 시작했다가 다시 시작: 짧은 줄 i 가 다음 줄의 앞부분과 같음
        if dup < 0 and i + 1 < len(lines) and len(norm[i]) >= 3 and lines[i + 1][0] - l[1] < 3:
            k = max(3, len(norm[i]) - 1)
            if norm[i + 1].startswith(norm[i][:k]) and len(norm[i + 1]) > len(norm[i]):
                dup, kind = i + 1, "restart"
        # 바로 이어서 같은 단어 반복 ("그 그", "저는 저는")
        ws = [_norm(w[2]) for w in l[3]]
        rep = [k for k in range(len(ws) - 1) if ws[k] and ws[k] == ws[k + 1]]
        out.append((dup, kind, rep))
    return out


# ---- 원고 파일 -------------------------------------------------------------
def script_path(mp3):
    return os.path.splitext(mp3)[0] + SCRIPT_SUFFIX


def html_path(mp3):
    return os.path.splitext(mp3)[0] + HTML_SUFFIX


def make_script(path, au=None):
    print(f"▶ 원고 만들기: {os.path.basename(path)}")
    au = au or Audio(path)
    ms = find_music_start(au)
    lines = split_lines(transcribe(path, vad=True))
    end_i, why = sermon_end_index(lines, ms, au.total)
    dups = find_duplicates(lines[:end_i + 1]) + [(-1, "", [])] * (len(lines) - end_i - 1)

    items = []
    for i, (s, e, t, w) in enumerate(lines):
        kind = "after" if i > end_i else ("callout" if is_callout(t) else "text")
        items.append({"s": round(s, 2), "e": round(e, 2), "t": t, "kind": kind,
                      "del": kind != "text",
                      "words": [[round(a, 2), round(b, 2), x.strip()] for a, b, x in w],
                      "dup": dups[i][0], "dupType": dups[i][1], "rep": dups[i][2]})

    sp = script_path(path)
    write_script(sp, path, items, [])
    n_call = sum(1 for r in items if r["kind"] == "callout")
    n_dup = sum(1 for r in items if r["dup"] >= 0)

    # 기본 상태의 예상 길이와 줄별 삭제 효과(대략)
    lines_rd, wcuts, _ = read_script(sp, quiet=True)
    base = plan(au, lines_rd, wcuts, verbose=False)[1]
    for it in items:
        it["cost"] = round((it["e"] - it["s"]) + 0.3, 2)

    hp = html_path(path)
    data = {"title": os.path.splitext(os.path.basename(path))[0], "mp3": os.path.basename(path),
            "mp3path": path, "scriptName": os.path.basename(sp), "created": int(time.time()),
            "base": round(base, 1), "maxLen": MAX_LEN_SEC, "lines": items}
    with open(hp, "w", encoding="utf-8") as f:
        f.write(render_page(data))

    print(f"  설교 끝: {tc(lines[end_i][1])}  ← {why}")
    print(f"  청중 호응 {n_call}곳 기본 삭제, 중복 의심 {n_dup}곳 표시")
    print(f"  기본 상태 예상 길이: {tc(base)}")
    print(f"✔ 원고: {sp}  ({len(items)}줄)")
    print(f"✔ 편집 화면: {hp}")
    return sp, hp


def _js(obj):
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_page(data, init=None):
    """편집 화면 HTML (init: 편집 서버가 넣어 주는 현재 상태)"""
    tpl = open(os.path.join(HERE, "editor_template.html"), encoding="utf-8").read()
    tpl = tpl.replace("/*DATA_BEGIN*/null/*DATA_END*/", "/*DATA_BEGIN*/" + _js(data) + "/*DATA_END*/")
    if init is not None:
        tpl = tpl.replace("/*INIT_BEGIN*/null/*INIT_END*/", "/*INIT_BEGIN*/" + _js(init) + "/*INIT_END*/")
    return tpl


def read_page_data(hp):
    h = open(hp, encoding="utf-8").read()
    m = re.search(r"/\*DATA_BEGIN\*/(.*?)/\*DATA_END\*/", h, re.S)
    if not m or m.group(1) == "null":
        return None
    return json.loads(m.group(1))


def state_from_script(data, sp):
    """원고.txt 의 # 표시와 단어 삭제를 편집 화면 상태(줄 번호 기준)로 바꿈"""
    lines, wcuts, _ = read_script(sp, quiet=True)
    by_start = {round(l[0], 2): l[3] for l in lines}
    cut_starts = {round(a, 2) for a, b in wcuts}
    dele = [by_start.get(round(L["s"], 2), L["del"]) for L in data["lines"]]
    cut = [[k for k, w in enumerate(L["words"]) if round(w[0], 2) in cut_starts] for L in data["lines"]]
    return {"del": dele, "cut": cut}


def write_script(sp, mp3, items, word_cuts):
    with open(sp, "w", encoding="utf-8-sig") as f:
        f.write(f"※ 극동방송 송출용 원고 - {os.path.splitext(os.path.basename(mp3))[0]}\n")
        f.write(f"※ 원본: {mp3}\n")
        f.write("※ 지울 문장은 줄 맨 앞에 # 을 붙이세요. 살릴 줄은 # 을 지우세요. (줄 자체를 지우지 마세요)\n")
        f.write("※ 글자 오타는 고쳐도 됩니다. [ ] 안의 시간만 그대로 두세요.\n")
        f.write("※ 청중 호응 멘트와 설교 이후(찬양 안내·찬양)는 기본으로 # 처리, 숨소리는 자동 삭제됩니다.\n")
        f.write(f"※ 총 {len(items)}줄\n\n")
        after = False
        for it in items:
            if it["kind"] == "after" and not after:
                f.write("\n======== 여기부터 설교 이후 (자동 삭제) ========\n")
                after = True
            t = it["t"] + ("  ← 청중 호응" if it["kind"] == "callout" else "")
            f.write(f"{'# ' if it['del'] else ''}[{tc2(it['s'])} ~ {tc2(it['e'])}] {t}\n")


def read_script(path, quiet=False):
    """([(start, end, text, deleted)], [(word_start, word_end)], 원본경로)"""
    lines, words, declared, src = [], [], None, None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            s_ = raw.strip()
            m = re.match(r"※ 총 (\d+)줄", s_)
            if m:
                declared = int(m.group(1)); continue
            if s_.startswith("※ 원본:"):
                src = s_.split(":", 1)[1].strip(); continue
            m = WORD_RE.match(raw)
            if m:
                words.append((int(m.group(1)) * 60 + float(m.group(2)), int(m.group(3)) * 60 + float(m.group(4))))
                continue
            m = LINE_RE.match(raw)
            if m:
                s = int(m.group(2)) * 60 + float(m.group(3))
                e = int(m.group(4)) * 60 + float(m.group(5))
                lines.append((s, e, m.group(6).strip(), bool(m.group(1))))
    if not quiet and declared is not None and declared != len(lines):
        print(f"  ⚠ 원고 줄 수가 {declared}줄 → {len(lines)}줄로 바뀌었습니다. 지워진 줄의 소리는 남아 있습니다.")
    return sorted(lines, key=lambda l: l[0]), sorted(words), src


# ---- 자를 위치 계산 --------------------------------------------------------
def cut_after(au, t_end, keep=JOIN_SEC):
    """t_end(말 끝) 바로 다음 쉼 안, 쉼 시작에서 keep초 뒤를 자를 위치로"""
    sil = au.silences(t_end - 0.4, t_end + 1.5)
    if not sil:
        return min(au.total, t_end + 0.05)
    s, e = sil[0]
    return min(s + keep, max(s, e - 0.05))


def cut_before(au, t_start, keep=JOIN_SEC):
    """t_start(말 시작) 바로 앞 쉼 안, 쉼 끝에서 keep초 앞을 다시 시작할 위치로"""
    sil = au.silences(t_start - 1.5, t_start + 0.4)
    if not sil:
        return max(0.0, t_start - 0.05)
    s, e = sil[-1]
    return max(e - keep, min(e, s + 0.05))


def intervals_from_script(au, lines):
    """남길 구간 [(t0, t1)] 과 지운 문장 목록"""
    blocks, cur = [], None
    for i, l in enumerate(lines):
        if l[3]:
            if cur: blocks.append(cur); cur = None
        else:
            cur = [i, i] if cur is None else [cur[0], i]
    if cur: blocks.append(cur)
    if not blocks:
        sys.exit("  ✖ 남길 문장이 없습니다.")
    iv = []
    for bi, (a, b) in enumerate(blocks):
        s, e = lines[a][0], lines[b][1]
        if a == 0:  # 맨 첫 문장부터 남기는 경우: 첫 말소리 1초 앞에서 시작
            first = next((x / 100 for x, y in runs(~au.quiet) if y - x >= 10 and x / 100 >= s - 0.5), s)
            t0 = max(0.0, first - LEAD_SEC)
        else:
            t0 = cut_before(au, s)
        t1 = cut_after(au, e, TAIL_SEC) if bi == len(blocks) - 1 else cut_after(au, e)
        iv.append((t0, t1))
    last_kept = lines[blocks[-1][1]][1]
    user_cuts = [l for l in lines if l[3] and l[1] <= last_kept]  # 설교 이후 자동 삭제 제외
    return iv, user_cuts


def word_cut_region(au, a, b):
    """단어 [a,b] 를 지울 실제 구간.
    음성인식 단어 시각은 앞뒤 쉼까지 늘어나 있는 경우가 많으므로, 단어 주변의 실제 쉼을 찾아
    그 쉼 안에서 자르고 양쪽 쉼은 WORD_JOIN_SEC 만큼만 남긴다. 쉼이 없으면 가장 조용한 지점에서 자른다."""
    def quietest(t0, t1):
        i0, i1 = max(0, int(t0 * 100)), min(len(au.db), int(t1 * 100))
        return (i0 + int(np.argmin(au.db[i0:i1]))) / 100 if i1 > i0 else t0

    pauses = [(s, e) for s, e in au.silences(a - 0.3, b + 0.5, 0.08)]
    left = [p for p in pauses if p[0] <= a + 0.6 and p[1] >= a - 0.15 and p[1] < b - 0.05]
    right = [p for p in pauses if p[1] >= b - 0.6 and p[0] <= b + 0.2 and p[0] > a + 0.05]
    if left:
        s, e = left[0]
        a2 = s + min(e - s, WORD_JOIN_SEC)
    else:
        a2 = quietest(a - 0.06, a + 0.06)
    right = [p for p in right if p[1] > a2]
    if right:
        s, e = right[-1]
        b2 = e - min(e - s, WORD_JOIN_SEC)
    else:
        b2 = quietest(b - 0.06, b + 0.06)
    return a2, max(a2, b2)


def subtract(iv, cuts):
    out = []
    for t0, t1 in iv:
        cur = [(t0, t1)]
        for a, b in cuts:
            nxt = []
            for x0, x1 in cur:
                if b <= x0 or a >= x1:
                    nxt.append((x0, x1)); continue
                if a > x0: nxt.append((x0, a))
                if b < x1: nxt.append((b, x1))
            cur = nxt
        out += [p for p in cur if p[1] - p[0] > 0.03]
    return out


# ---- 편집 ----------------------------------------------------------------
def sentence_bounds(lines):
    """문장이 끝나고 다음 문장이 시작하는 자리 [(앞 문장 끝, 다음 문장 시작)]"""
    kept = [l for l in lines if not l[3]]
    out = []
    for a, b in zip(kept[:-1], kept[1:]):
        t = re.sub(r"\s*←.*$", "", a[2]).strip()
        if SENT_END.search(t):
            out.append((a[1], b[0]))
    return out


def pause_target(L, sentence):
    if sentence:  # 문장 사이: 넉넉히
        return min(L, max(SENT_MIN, min(SENT_KEEP * L, SENT_MAX)))
    return min(L, max(0.2, min(PAUSE_KEEP * L, PAUSE_MAX)))


def shorten_pauses(au, t0, t1, bounds=()):
    """구간 [t0, t1] 안의 긴 쉼을 줄인 구간 목록. 숨소리는 쉼으로 보고(소리만 줄임) 시간은 그대로 계산."""
    out, cur, n, rem = [], t0, 0, 0.0
    for s, e in au.silences(t0, t1, MIN_PAUSE):
        if s <= t0 + 0.01 or e >= t1 - 0.01:
            continue
        sentence = any(s < bs + 0.4 and e > be - 0.4 for be, bs in bounds)
        half = pause_target(e - s, sentence) / 2
        c0, c1 = s + half, e - half
        if c1 - c0 < 0.02:
            continue
        out.append((cur, c0)); cur = c1; n += 1; rem += c1 - c0
    out.append((cur, t1))
    return out, n, rem


def plan(au, lines, word_cuts, verbose=True):
    """남길 조각 목록과 예상 길이"""
    iv, user_cuts = intervals_from_script(au, lines)
    if verbose:
        kept = [l for l in lines if not l[3]]
        print(f"  설교 끝: '{kept[-1][2]}' ({tc(kept[-1][1])})")
        if user_cuts:
            print(f"  지운 문장 {len(user_cuts)}줄:")
            for s, e, t, _ in user_cuts:
                print(f"    ✂ {tc(s)}  {t}")
    regions = [word_cut_region(au, a, b) for a, b in word_cuts]
    if verbose and word_cuts:
        print(f"  지운 단어 {len(word_cuts)}개")
    iv = subtract(iv, regions)
    au.drop[:] = False
    nb = 0
    if REMOVE_BREATH:
        for t0, t1 in iv:
            for a, b in detect_breaths(au, t0, t1):
                au.mark_drop(a, b); nb += 1
    bounds = sentence_bounds(lines)
    pieces, n, rem = [], 0, 0.0
    for t0, t1 in iv:
        p, k, r = shorten_pauses(au, t0, t1, bounds)
        pieces += p; n += k; rem += r
    if verbose:
        print(f"  숨소리 {nb}곳 소리 제거, 쉼 {n}곳 줄임 (총 {rem:.1f}초, 문장 사이 쉼은 넉넉히)")
    return pieces, sum(b - a for a, b in pieces) + PAD_START + PAD_END


def stitch(au, pieces_sec):
    x = au.x
    xf = int(XFADE * SR)
    ramp = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
    pieces = []
    for a, b in pieces_sec:
        s, e = int(a * SR), int(b * SR)
        if e - s <= xf:
            continue
        if not pieces:
            pieces.append(x[s:e] * au.pause_gain(s, e)[:, None]); continue
        s0 = max(0, s - xf)
        seg = x[s0:e] * au.pause_gain(s0, e)[:, None]
        prev = pieces[-1]
        prev[-xf:] = prev[-xf:] * (1 - ramp) + seg[:xf] * ramp
        pieces.append(seg[xf:])
    y = np.concatenate(pieces)
    nf = int(FADE_OUT * SR)
    y[-nf:] *= np.linspace(1, 0, nf, dtype=np.float32)[:, None]
    z = lambda sec: np.zeros((int(sec * SR), 2), np.float32)
    return np.concatenate([z(PAD_START), y, z(PAD_END)])


def build(mp3, out, sp):
    print(f"▶ 송출 파일 만들기: {os.path.basename(mp3)}")
    au = Audio(mp3)
    print(f"  원본 길이 {tc(au.total)}")
    if not os.path.exists(sp):
        print("  원고가 없어 먼저 만듭니다 (기본값으로 편집)")
        sp, _ = make_script(mp3, au)
    print(f"  원고 사용: {sp}")
    lines, word_cuts, _ = read_script(sp)
    pieces, _ = plan(au, lines, word_cuts)
    y = stitch(au, pieces)

    with tempfile.TemporaryDirectory() as td:
        raw, comp = os.path.join(td, "cut.f32"), os.path.join(td, "comp.wav")
        y.astype(np.float32).tofile(raw)
        pre = REF_IN_LUFS - loudness(raw, ["-f", "f32le", "-ar", str(SR), "-ac", "2"])["I"]
        nz = f"{DENOISE}," if NOISE_MODE == "denoise" else ""
        print(f"  녹음 크기 보정 {pre:+.1f} dB, 잡음 처리: " +
              ("전체 잡음 제거" if nz else f"쉼 구간 -{PAUSE_ATTEN_DB:.0f} dB"))
        run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", raw,
             "-af", f"volume={pre:.2f}dB,{nz}{COMPRESSOR}", "-c:a", "pcm_f32le", comp])
        gain = TARGET_LUFS - loudness(comp)["I"]
        lim = 10 ** (LIMIT_DB / 20)
        run([FF, "-v", "error", "-y", "-i", comp, "-af",
             f"volume={gain:.2f}dB,alimiter=limit={lim:.4f}:attack=5:release=50:level=disabled",
             "-c:a", "libmp3lame", "-b:a", "192k", "-ar", str(SR), "-ac", "2", out])
    r = loudness(out)
    dur = len(y) / SR
    print(f"✔ 완성: {out}")
    print(f"  길이 {tc(dur)} (원본 대비 -{tc(au.total - dur)}),  {r['I']} LUFS, LRA {r['LRA']} LU, 피크 {r['TP']} dBTP")
    if dur > MAX_LEN_SEC:
        print(f"  ⚠ 경고: 송출 기준 25분을 {tc(dur - MAX_LEN_SEC)} 초과합니다. 원고에서 더 지워 주세요.")
    else:
        print(f"  ✔ 송출 기준 25분 이내 (여유 {tc(MAX_LEN_SEC - dur)})")


def loudness(p, in_args=()):
    r = subprocess.run([FF, "-hide_banner", "-nostats", *in_args, "-i", p, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    t = r[r.rfind("Summary:"):]
    g = lambda k: float(re.search(k + r":\s+(-?[\d.]+)", t).group(1))
    return {"I": g("I"), "LRA": g("LRA"), "TP": g("Peak")}


# ---- 편집 서버 (이 PC 안에서만 접속: 127.0.0.1) ----------------------------
EDIT_PORT = 8731


class _Tee:
    """build() 의 출력을 콘솔과 편집 화면(진행 상황) 양쪽으로"""
    def __init__(self, console):
        self.console, self.buf = console, []

    def write(self, s):
        self.buf.append(s)
        try:
            self.console.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass


def serve(mp3, sp, open_browser=True):
    import threading, webbrowser, contextlib, traceback
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    hp = html_path(mp3)
    data = read_page_data(hp) if os.path.exists(hp) else None
    if data is None:
        if os.path.exists(sp):
            print("  기존 원고는 편집 화면 정보가 없어 새로 만듭니다 (기존 원고는 '_원고_이전.txt'로 보관)")
            bak, k = sp[:-4] + "_이전.txt", 2
            while os.path.exists(bak):
                bak, k = sp[:-4] + f"_이전{k}.txt", k + 1
            os.rename(sp, bak)
        sp, hp = make_script(mp3)
        data = read_page_data(hp)
    elif not os.path.exists(sp):
        print("  원고 파일이 없어 기본 상태로 다시 만듭니다")
        write_script(sp, mp3, data["lines"], [])
    job = {"tee": None, "done": True, "ok": False, "out": ""}
    lock = threading.Lock()

    def do_build(text):
        tee = job["tee"]
        try:
            with open(sp, "w", encoding="utf-8-sig", newline="") as f:
                f.write(text.lstrip("﻿"))
            out = default_out(mp3)
            with contextlib.redirect_stdout(tee):
                build(mp3, out, sp)
            job.update(ok=True, out=os.path.basename(out))
        except BaseException:
            tee.write("\n✖ 오류:\n" + traceback.format_exc())
        finally:
            job["done"] = True
            lock.release()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20_000_000:
                raise ValueError("too large")
            return self.rfile.read(n).decode("utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(200, render_page(data, state_from_script(data, sp)), "text/html; charset=utf-8")
                elif u.path == "/audio":
                    self._audio()
                elif u.path == "/progress":
                    pos = int(parse_qs(u.query).get("from", ["0"])[0])
                    tee = job["tee"]
                    text = "".join(tee.buf) if tee else ""
                    self._send(200, json.dumps({"text": text[pos:], "pos": len(text), "done": job["done"],
                                                "ok": job["ok"], "out": job["out"]}, ensure_ascii=False),
                               "application/json; charset=utf-8")
                else:
                    self._send(404, "not found")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _audio(self):
            size = os.path.getsize(mp3)
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(mp3, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                text = self._body()
                if "[" not in text:
                    return self._send(400, "원고 내용이 비어 있습니다")
                if u.path == "/save":
                    if not job["done"]:
                        return self._send(409, "송출 파일을 만드는 중입니다")
                    with open(sp, "w", encoding="utf-8-sig", newline="") as f:
                        f.write(text.lstrip("﻿"))
                    self._send(200, "ok")
                elif u.path == "/build":
                    if not lock.acquire(blocking=False):
                        return self._send(409, "이미 송출 파일을 만드는 중입니다")
                    job.update(done=False, ok=False, out="", tee=_Tee(sys.__stdout__))
                    threading.Thread(target=do_build, args=(text,), daemon=True).start()
                    self._send(200, "started")
                else:
                    self._send(404, "not found")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    srv = None
    for port in range(EDIT_PORT, EDIT_PORT + 10):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            continue
    if srv is None:
        sys.exit("✖ 편집 서버를 열 수 없습니다 (포트 사용 중)")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print("")
    print("━" * 60)
    print(f"  편집 화면: {url}")
    print("  브라우저에서 편집하고 [송출 파일 만들기]를 누르세요.")
    print("  ※ 이 창을 닫으면 편집 화면이 멈춥니다. 다 끝나면 닫으세요.")
    print("━" * 60)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def resolve(p):
    """입력(MP3 또는 원고.txt) -> (원본 mp3, 원고 경로)"""
    p = os.path.abspath(p)
    if p.lower().endswith(".txt"):
        _, _, src = read_script(p, quiet=True)
        mp3 = src if src and os.path.exists(src) else re.sub(r"_원고( \(\d+\))?\.txt$", ".mp3", p)
        if not os.path.exists(mp3):
            sys.exit(f"✖ 원본 MP3를 찾을 수 없습니다: {mp3}")
        return mp3, p
    if not os.path.exists(p):
        sys.exit(f"✖ 파일을 찾을 수 없습니다: {p}")
    return p, script_path(p)


def default_out(path):
    stem, _ = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}_{i}.mp3"):
        i += 1
    return f"{stem}_{i}.mp3"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--edit", action="store_true", help="편집 화면 열기 (원고가 없으면 먼저 만듦) + 송출 파일 만들기 버튼")
    ap.add_argument("--script", action="store_true", help="원고(_원고.txt)와 편집 화면(_편집.html) 만들기")
    ap.add_argument("--force", action="store_true", help="원고가 있어도 새로 만들기")
    ap.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    if not a.inputs:
        print("원본 설교 MP3 파일을 '설교편집' 아이콘 위로 끌어다 놓으세요.")
        print(" - 처음이면 받아쓰기(약 5분) 후 편집 화면이 열립니다.")
        print(" - 편집하던 설교면 저장된 상태 그대로 편집 화면이 열립니다.")
        print(" - 편집 화면에서 [송출 파일 만들기]를 누르면 같은 폴더에 _1.mp3 가 만들어집니다.")
        sys.exit(0)
    if a.edit:
        os.system("title 극동방송 설교 편집 - 이 창을 닫으면 편집 화면이 멈춥니다")
    for p in a.inputs:
        mp3, sp = resolve(p)
        if a.edit:
            serve(mp3, sp, not a.no_browser)
            break
        if a.script:
            if os.path.exists(sp) and not a.force:
                print(f"✖ 원고가 이미 있습니다 (덮어쓰지 않음): {sp}")
                print("  편집 화면만 다시 열려면 _편집.html 을 더블클릭하세요.")
            else:
                make_script(mp3)
        else:
            build(mp3, a.out or default_out(mp3), sp)
        print()
