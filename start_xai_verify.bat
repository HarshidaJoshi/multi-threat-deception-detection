@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   XAI-Verify (Django + XAMPP MySQL)
echo ============================================

if not exist .env (
  echo Creating .env from .env.example...
  copy .env.example .env
)

echo.
echo [XAMPP] Start MySQL in XAMPP Control Panel before continuing.
echo         Create database "xai_verify" in phpMyAdmin if not done yet.
echo         See setup_xampp.sql or README.md
echo.

echo [1/3] Installing Python dependencies...
if exist venv\Scripts\pip.exe (
  venv\Scripts\pip install -r requirements.txt -q
  set PY=venv\Scripts\python.exe
) else (
  pip install -r requirements.txt -q
  set PY=python
)

echo.
echo [2/3] Database migrations...
%PY% manage.py migrate
if errorlevel 1 (
  echo.
  echo MIGRATION FAILED — check that XAMPP MySQL is running and database xai_verify exists.
  pause
  exit /b 1
)

echo.
echo [3/3] Starting Django on http://127.0.0.1:8000
echo   Press Ctrl+C to stop.
echo.
%PY% manage.py runserver 127.0.0.1:8000
