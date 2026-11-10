@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env
echo Start XAMPP MySQL before running migrations.
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe manage.py migrate
  venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
) else (
  python manage.py migrate
  python manage.py runserver 127.0.0.1:8000
)
