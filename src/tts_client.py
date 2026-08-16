# -*- coding: utf-8 -*-
"""tts_client.py — GUI(메인 파이썬)에서 tts_server(sbv2_env)를 띄우고 호출.

llm_server가 llama-server를 관리하는 것과 같은 구도. 무거운 torch/SBV2는 별도
venv의 별도 프로세스(tts_server.py)가 지고, GUI는 이 얇은 클라이언트로 HTTP만 친다.
동봉/개발 환경에 sbv2_env가 없으면 available()=False → GUI가 조용히 무음 폴백.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / "tts" / "sbv2_env" / "Scripts" / "python.exe"
SERVER = ROOT / "src" / "tts_server.py"
PORT = 8848

_proc: subprocess.Popen | None = None
_job = None


def base_url() -> str:
    return f"http://127.0.0.1:{PORT}"


def available() -> bool:
    """sbv2_env + 서버 스크립트가 존재하는가(실행 여부와 무관)."""
    return VENV_PY.exists() and SERVER.exists()


def _health(timeout: float = 1.0) -> bool:
    try:
        r = urllib.request.urlopen(base_url() + "/health", timeout=timeout)
        return r.status == 200
    except Exception:
        return False


def ready() -> bool:
    return _health()


def _make_job():
    """kill-on-close Job Object (llm_server와 동일): 부모가 죽으면 자식도 정리."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IOC),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = EXT()
    info.BasicLimitInformation.LimitFlags = 0x2000   # KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None
    return job


def _launch() -> subprocess.Popen | None:
    global _job
    if not available():
        return None
    # TTS는 CPU로 — GPU는 LLM(llama-server) 전용으로 두어 메모리 경합/다운을 막는다.
    args = [str(VENV_PY), str(SERVER), "--port", str(PORT),
            "--device", "cpu", "--preload", "KO"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        p = subprocess.Popen(args, cwd=str(ROOT), creationflags=flags,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if os.name == "nt":
        if _job is None:
            _job = _make_job()
        if _job:
            import ctypes
            ctypes.windll.kernel32.AssignProcessToJobObject(_job, int(p._handle))
    return p


def ensure() -> str | None:
    """서버 실행 보장(로딩 완료 대기 안 함 — ready()로 확인)."""
    global _proc
    if not available():
        return None
    if _health():
        return base_url()
    if _proc is not None and _proc.poll() is None:
        return base_url()
    _proc = _launch()
    return base_url() if _proc else None


def synth(text: str, emotion: str = "neutral", lang: str = "KO",
          timeout: float = 30.0, warmup_wait: float = 55.0) -> bytes | None:
    """텍스트+감정 -> wav bytes. 실패 시 None(무음 폴백).

    서버가 아직 모델 로딩 중이면(첫 실행 ~24초) 연결이 거부된다. 그동안
    warmup_wait 초까지 재시도해서, 준비되는 즉시 소리가 나오게 한다."""
    if not text.strip():
        return None
    ensure()                                     # 서버 없으면 띄우기(idempotent)
    body = json.dumps({"text": text, "emotion": emotion, "lang": lang}).encode("utf-8")
    deadline = time.time() + warmup_wait
    while True:
        req = urllib.request.Request(base_url() + "/synth", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.read() if r.status == 200 else None
        except urllib.error.URLError:            # 서버 로딩 중(연결 거부) — 잠깐 뒤 재시도
            if time.time() >= deadline:
                return None
            time.sleep(1.5)
        except Exception:
            return None
