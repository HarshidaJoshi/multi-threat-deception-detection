@echo off
rem Resilient launcher: re-runs the (resumable) job until it writes its results file, at most 20 attempts.
cd /d F:\deepfake_detection_project
if exist benchmark_outull_frozenesults_full_frozen.json exit /b 0
set TRIES=0
:loop
set /a TRIES+=1
echo [launcher] attempt %TRIES% at %TIME% >> full_frozen.log
venv\Scripts\python.exe scripts\train_full_frozen_features.py >> full_frozen.log 2>> full_frozen.err
if exist benchmark_out\full_frozen\results_full_frozen.json goto done
if %TRIES% GEQ 20 goto done
timeout /t 15 /nobreak > nul
goto loop
:done
echo [launcher] finished at %TIME% >> full_frozen.log
