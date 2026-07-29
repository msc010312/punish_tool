@echo off
REM Build distributable exe. Move-ID (torch/DINOv2/YOLO) excluded on purpose.
REM Report provides: counter/punish detection, damage, damage-based starter
REM candidates, combos. Bundled Tesseract for banner/char-name OCR.
REM To re-enable move-ID later: restore rag_index.npz + torch_hub, set PUNISH_MOVEID=1

python -m PyInstaller --noconfirm --windowed --name FrameAnalyzer ^
  --icon app.ico ^
  --paths src ^
  --hidden-import pytesseract ^
  --hidden-import hud_reader --hidden-import punish_engine --hidden-import analyze_match ^
  --hidden-import move_names --hidden-import banner_match --hidden-import reversal_wake ^
  --hidden-import banner_ncc ^
  --hidden-import input_reader --hidden-import command_parser --hidden-import input_moveid ^
  --hidden-import coach --hidden-import coach_kb --hidden-import coach_web ^
  --hidden-import llm_server --hidden-import avatar --hidden-import avatar3d ^
  --hidden-import PySide6.QtWebEngineWidgets --hidden-import PySide6.QtWebEngineCore ^
  --hidden-import games --hidden-import games.ggst --hidden-import games.profile ^
  --exclude-module torch --exclude-module torchvision --exclude-module ultralytics ^
  --exclude-module transformers --exclude-module bitsandbytes --exclude-module peft ^
  --exclude-module qwen_vl_utils --exclude-module rag_id --exclude-module vlm_id ^
  src\gui.py

if errorlevel 1 goto :fail

set D=dist\FrameAnalyzer

REM data json
copy /Y framedata_ggst.json      %D%\
copy /Y combos_ggst.json         %D%\
copy /Y mechanics_ggst.json      %D%\
copy /Y char_notes_ggst.json     %D%\
copy /Y move_names_ko.json       %D%\
copy /Y starter_prior_ggst.json  %D%\
copy /Y char_stats_ggst.json     %D%\
copy /Y hud_config.json          %D%\

REM banner templates (recovers banners Tesseract misses)
copy /Y banner_templates.npz     %D%\

REM white (wakeup) REVERSAL template - torch-free NCC detector
copy /Y reversal_wake_tpl.npz    %D%\

REM banner NCC templates (fallback to recover banners OCR misses)
copy /Y banner_templates_ncc.npz %D%\

REM resources
copy /Y app.ico                  %D%\
copy /Y "Impact - Strive.otf"    %D%\

REM bundled Tesseract (app finds it next to the exe)
if exist tesseract\tesseract.exe (
  xcopy /E /I /Y tesseract %D%\tesseract >nul
  echo [OK] Tesseract bundled
) else (
  echo [WARN] no tesseract\ folder - OCR will be missing in the build
)

REM coach avatar sprites (drop-in folder; empty = avatar hidden)
if exist avatar (
  xcopy /E /I /Y avatar %D%\avatar >nul
  echo [OK] avatar folder bundled
)

REM realtime 3D avatar (three.js + GLB; missing = falls back to sprites)
if exist avatar3d\sol.glb (
  xcopy /E /I /Y avatar3d %D%\avatar3d >nul
  echo [OK] avatar3d bundled
)

REM bundled llama.cpp + GGUF model (AI coach; no Ollama needed).
REM /D = copy only newer (the 5GB model is skipped on rebuilds)
if exist llama\llama-server.exe (
  xcopy /E /I /Y /D llama %D%\llama >nul
  echo [OK] llama.cpp + model bundled
) else (
  echo [WARN] no llama\ folder - AI coach will be facts-only
)

del %D%\timeline.json 2>nul
echo.
echo [DONE] %D%\FrameAnalyzer.exe
goto :eof

:fail
echo.
echo [FAIL] PyInstaller failed. See output above.
exit /b 1
