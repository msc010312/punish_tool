# -*- coding: utf-8 -*-
"""tts_server.py — 솔 음성 로컬 합성 서버 (sbv2_env에서 실행).

코치 답변 텍스트 + 감정 라벨을 받아 솔 목소리 wav를 돌려준다. SBV2 모델을
1회만 로드하고 상주한다. GUI(메인 파이썬)가 이 서버(127.0.0.1)로 POST 한다.
llm_server가 llama-server를 띄우는 것과 같은 구도 — GUI는 가볍게 두고 무거운
torch/SBV2는 이 별도 프로세스(별도 venv)가 진다.

감정은 스타일(Battle 전투톤/Story 차분톤) + 속도 + sdp로만 준다.
intonation_scale/pitch_scale은 pyworld 후처리를 켜서 기계음을 유발하므로 절대
1.0에서 벗어나지 않는다(음성 확정 과정에서 검증됨). noise_w=0.5는 ㅆ 번짐 완화값.

엔드포인트
  GET  /health        -> 200 "ok"(모델 로드 완료 시)
  POST /synth {text, emotion, lang}  -> audio/wav

사용: tts/sbv2_env/Scripts/python.exe src/tts_server.py --port 8848
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBV2 = ROOT / "tts" / "sbv2"
sys.path.insert(0, str(SBV2))

# 감정 프리셋: emote 라벨 -> (style, style_weight, length, sdp_ratio)
# 라벨은 gui._pick_emote 출력(joy/shake/angry/smirk/sad/sigh/surprised/nod)과 일치.
EMOTION_PRESETS = {
    "neutral":   ("Neutral", 1.0, 1.00, 0.20),
    "angry":     ("Battle",  1.6, 0.93, 0.25),
    "smirk":     ("Battle",  1.4, 1.00, 0.35),   # 도발
    "joy":       ("Battle",  1.4, 0.98, 0.28),   # 만족·칭찬
    "surprised": ("Battle",  1.5, 0.96, 0.30),
    "sad":       ("Story",   1.6, 1.08, 0.20),   # 아쉬움
    "sigh":      ("Story",   1.6, 1.10, 0.18),   # 은은한 한숨(문장에 녹임)
    "shake":     ("Story",   1.3, 1.00, 0.20),   # 부정·단호
    "nod":       ("Neutral", 1.1, 1.00, 0.22),   # 수긍
}
BASE = dict(noise=0.6, noise_w=0.5, intonation_scale=1.0, pitch_scale=1.0)

# 언어별 모델 경로(체크포인트는 최신 또는 지정). KO=감정스타일 보유, JP=Neutral만.
MODEL_SPEC = {
    "KO": (SBV2 / "Data/SOL", SBV2 / "Data/SOL/models_e70base_snap/G_11400_e110.pth"),
    "JP": (SBV2 / "Data/SOL_JP", None),   # None=최신 체크포인트
}

_models = {}          # lang -> TTSModel
_Languages = None


def _shims():
    import torch, torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})


def _pth_to_safetensors(pth: Path) -> Path:
    import torch
    from safetensors.torch import save_file
    out = pth.with_suffix(".safetensors")
    if not out.exists():
        ck = torch.load(str(pth), map_location="cpu")
        sd = ck["model"] if "model" in ck else ck
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out))
    return out


def _latest_ckpt(models_dir: Path) -> Path:
    cks = sorted(models_dir.glob("G_[0-9]*.pth"),
                 key=lambda p: int(p.stem.split("_")[1]))
    if not cks:
        raise FileNotFoundError(f"체크포인트 없음: {models_dir}")
    return cks[-1]


def get_model(lang: str, device: str):
    global _Languages
    if lang in _models:
        return _models[lang]
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
    _Languages = Languages
    mdir, ckpt = MODEL_SPEC[lang]
    if ckpt is None:
        ckpt = _latest_ckpt(mdir / "models")
    safet = _pth_to_safetensors(ckpt)
    m = TTSModel(model_path=safet, config_path=mdir / "config.json",
                 style_vec_path=mdir / "style_vectors.npy", device=device)
    m.load()
    _models[lang] = m
    print(f"[tts_server] {lang} 모델 로드: {safet.name}", flush=True)
    return m


def synth_wav(text: str, emotion: str, lang: str, device: str) -> bytes:
    import soundfile as sf
    model = get_model(lang, device)
    style, weight, length, sdp = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])
    # 이 모델에 없는 스타일이면 중립으로 대체(예: JP엔 Story가 없음 → 차분한 감정은 Neutral).
    avail = getattr(model, "style2id", {"Neutral": 0})
    if style not in avail:
        style, weight = "Neutral", 1.0
    sr, audio = model.infer(text=text, language=_Languages[lang], speaker_id=0,
                            style=style, style_weight=weight, length=length,
                            sdp_ratio=sdp, **BASE)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    device = "cuda"

    def log_message(self, *a):        # 조용히
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/synth":
            self._send(404, b"not found", "text/plain"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = (body.get("text") or "").strip()
            emotion = body.get("emotion") or "neutral"
            lang = (body.get("lang") or "KO").upper()
            if not text:
                self._send(400, b"empty text", "text/plain"); return
            wav = synth_wav(text, emotion, lang, self.device)
            self._send(200, wav, "audio/wav")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, str(e).encode("utf-8"), "text/plain")

    def _send(self, code: int, data: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preload", default="KO", help="시작 시 미리 로드할 언어(빈값=지연로드)")
    args = ap.parse_args()

    _shims()
    Handler.device = args.device
    if args.preload:
        try:
            get_model(args.preload, args.device)
        except Exception as e:
            print(f"[tts_server] preload 실패({args.preload}): {e}", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[tts_server] listening on http://127.0.0.1:{args.port} (device={args.device})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
