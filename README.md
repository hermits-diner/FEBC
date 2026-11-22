# FEBC 설교 편집기

극동방송 송출용 설교 MP3 편집 도구입니다.

## 편집 규칙
- 앞: 첫 말소리 1초 전부터 시작
- 끝: 설교 마지막 "…축복합니다" 이후의 찬양·기도 안내와 찬양 삭제
- 청중 호응 멘트("아멘이십니까?", "할렐루야") 기본 삭제, 숨소리 자동 처리
- 긴 쉼 줄이기, 음량 -25.5 LUFS
- MP3 192kbps / 48kHz / 스테레오, **25분 이내**

## 파일
| 파일 | 설명 |
|---|---|
| `febc_edit.py` | 받아쓰기·원고 생성·편집 서버·송출 파일 생성 |
| `editor_template.html` | 브라우저 편집 화면 템플릿 |
| `gemini_review.py` | Gemini 편집 제안 (선택) |
| `설교편집.bat` | MP3를 끌어다 놓으면 편집 화면 실행 |

## 설치
```
pip install numpy imageio-ffmpeg faster-whisper google-genai
```
Gemini 제안을 쓰려면 `.env.example` 을 `.env` 로 복사하고 `GEMINI_API_KEY` 를 넣으세요.

## 사용법
1. 원본 설교 MP3를 `설교편집.bat` 위로 끌어다 놓기
2. 브라우저 편집 화면에서 지울 줄/단어 선택
3. **[송출 파일 만들기]** → 같은 폴더에 `원본_1.mp3` 생성

명령줄:
```
python febc_edit.py --edit "원본.mp3"     # 편집 화면
python febc_edit.py --script "원본.mp3"   # 원고만 만들기
python febc_edit.py "원본.mp3"            # 바로 송출 파일 만들기
```
