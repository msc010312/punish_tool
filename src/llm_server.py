# -*- coding: utf-8 -*-
"""llm_server.py — 동봉된 llama.cpp(llama-server) 자동 실행/관리.

배포 원칙(설치 없음·터미널 없음) 준수: 유저는 Ollama를 몰라도 된다.
앱 폴더의 llama/llama-server.exe + *.gguf 를 찾아 백그라운드로 띄우고
OpenAI 호환 API(127.0.0.1)를 coach.py 에 제공한다.

  ensure()  -> base_url|None   서버 준비(없으면 실행, 로딩 대기는 안 함)
  ready()   -> bool            모델 로딩까지 끝나 응답 가능한가
  base_url()-> str             http://127.0.0.1:<port>
  stop()                       앱 종료 시 프로세스 정리(atexit 자동)

GPU: Vulkan 빌드( NVIDIA/AMD/인텔 공통 ) -ngl 99 로 시도, 30초 내 헬스체크
실패 시 CPU(-ngl 0)로 1회 재시도. 저사양이어도 느릴 뿐 동작은 한다.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import time
import urllib.request

PORT = int(os.environ.get("PUNISH_LLM_PORT", "18321"))
_proc: subprocess.Popen | None = None
_gpu_failed = False          # GPU 실패 기록(재시도 시 CPU로)
_job = None                  # Windows Job Object 핸들(부모 죽으면 자식도 죽게)


def _make_job():
    """kill-on-close Job Object. 앱이 '강제 종료/크래시'돼도(atexit 안 돎)
    OS가 핸들을 닫으면서 자식 llama-server 를 같이 정리한다."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9,       # JobObjectExtendedLimitInformation
                                       ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None
    return job


def base_url() -> str:
    return f"http://127.0.0.1:{PORT}"


def _paths():
    """(server_exe, model_gguf) — 없으면 (None, None). 동봉 폴더 = app_dir/llama."""
    import punish_engine as pe
    d = pe.app_dir() / "llama"
    exe = d / "llama-server.exe"
    if not exe.exists():
        return None, None
    ggufs = sorted(d.glob("*.gguf"))
    if not ggufs:
        return None, None
    return exe, ggufs[0]


def available() -> bool:
    """동봉 서버+모델 파일이 존재하는가(실행 여부와 무관)."""
    exe, gguf = _paths()
    return exe is not None and gguf is not None


def _health(timeout: float = 1.0) -> bool:
    """모델 로딩 완료 시 /health 가 200. 로딩 중엔 503."""
    try:
        r = urllib.request.urlopen(base_url() + "/health", timeout=timeout)
        return r.status == 200
    except Exception:
        return False


def ready() -> bool:
    return _health()


def _launch(ngl: int) -> subprocess.Popen | None:
    global _job
    exe, gguf = _paths()
    if exe is None:
        return None
    args = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT),
            "-ngl", str(ngl), "-c", "8192", "--jinja", "--no-webui"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        p = subprocess.Popen(args, creationflags=flags,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if os.name == "nt":                    # 부모 사망 시 자식 자동 정리
        if _job is None:
            _job = _make_job()
        if _job:
            import ctypes
            ctypes.windll.kernel32.AssignProcessToJobObject(_job, int(p._handle))
    return p


def ensure() -> str | None:
    """서버 실행 보장(로딩 완료 대기는 안 함 — ready()로 확인).
    반환: base_url, 동봉 파일 없으면 None."""
    global _proc, _gpu_failed
    if not available():
        return None
    if _health():
        return base_url()
    if _proc is not None and _proc.poll() is None:
        return base_url()                      # 실행 중(로딩 중일 수 있음)
    _proc = _launch(0 if _gpu_failed else 99)
    return base_url() if _proc else None


def wait_ready(timeout: float = 120.0) -> bool:
    """로딩 완료까지 대기. GPU 시도가 죽으면 CPU 로 1회 재시도."""
    global _proc, _gpu_failed
    if ensure() is None:
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _health():
            return True
        if _proc is not None and _proc.poll() is not None and not _gpu_failed:
            _gpu_failed = True                 # GPU(-ngl 99) 실패 -> CPU 재시도
            _proc = _launch(0)
            if _proc is None:
                return False
        time.sleep(1.0)
    return False


def stop():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


atexit.register(stop)
