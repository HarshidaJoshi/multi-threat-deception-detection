# Installation guide

## Prerequisites

- Windows 10/11
- Python 3.10+
- XAMPP (MySQL)
- 8 GB+ RAM recommended (TensorFlow + BERT)

## Steps

### 1. Clone / open project

```powershell
cd "d:\MYDATA\Desktop\MTECH\3rd SEM\deepfake_detection_project"
```

### 2. Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment file

```powershell
copy .env.example .env
```

Edit `.env` if MySQL credentials differ from XAMPP defaults.

### 4. MySQL database

1. Start **MySQL** in XAMPP Control Panel
2. Open phpMyAdmin → create database `xai_verify`
3. Or run `setup_xampp.sql` in MySQL

### 5. Migrations

```powershell
.\venv\Scripts\python manage.py migrate
```

### 6. (Optional) Superuser

```powershell
.\venv\Scripts\python manage.py createsuperuser
```

### 7. Train deepfake model (optional)

```powershell
# One command (seed if needed + train)
.\venv\Scripts\python manage.py train_demo_deepfake --dataset datasets
```

Or step by step:

```powershell
.\venv\Scripts\python manage.py seed_demo_dataset --dataset datasets
.\venv\Scripts\python manage.py train_deepfake --dataset datasets
```

### 8. Run server

```powershell
.\start_xai_verify.bat
```

Visit http://127.0.0.1:8000

## Demo mode

Set `XAI_DEMO_MODE=1` in `.env` to skip MySQL writes and store history in session only.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `migrate` fails | Ensure XAMPP MySQL is running and `xai_verify` DB exists |
| BERT slow first run | HuggingFace downloads model on first misinformation analysis |
| Deepfake uses forensics only | Run `python manage.py train_demo_deepfake --dataset datasets` |
| TensorFlow errors | Use Python 3.10–3.11; check `pip install tensorflow` |
