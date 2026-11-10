@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Training BERT and RoBERTa on WELFake (use GPU if available)...
echo For a quick test: python scripts\train_fake_news_transformers.py --smoke
python scripts\train_fake_news_transformers.py --architecture both
pause
