# XAI-Verify

**A Multi-Threat Explainable AI Framework for Digital Deception Detection and Evidence-Aware Authenticity Verification**

M.Tech dissertation research prototype — Django + MySQL (XAMPP).

## Research modules

| Module | Description | XAI methods |
|--------|-------------|-------------|
| **DeepFake Image Detection** | EfficientNetB0 on real vs. AI-generated images | Grad-CAM heatmaps |
| **Misinformation Detection** | Text analysis of misleading content | LIME tokens, SHAP features |
| ↳ Fake News | BERT fake-news classifier | LIME, SHAP |
| ↳ Recruitment Scam | Application to fraudulent job posts | LIME, SHAP |
| **Analysis History** | Stored results + PDF reports | — |

## Project structure

```
deepfake_detection_project/
├── manage.py
├── requirements.txt
├── start_xai_verify.bat
├── .env
│
├── xai_verify/              # Django settings & root URLs (configs)
├── apps/
│   ├── dashboard/           # Home & about pages
│   ├── deepfake/            # Image upload UI
│   ├── misinformation/      # Fake news + recruitment scam
│   ├── explainability/      # XAI service layer (Grad-CAM, LIME, SHAP)
│   ├── reports/             # History & PDF downloads
│   └── accounts/            # Login / logout
├── utils/                   # ML pipelines (TensorFlow, BERT, orchestration)
├── templates/
├── static/
├── datasets/                # Training images (real/ + fake/)
├── trained_models/          # Saved weights after training
├── media/                   # Runtime uploads
├── generated_reports/
└── docs/                    # Installation, API, deployment, architecture
```

## Quick start

1. Start XAMPP MySQL, create database `xai_verify` (`setup_xampp.sql`)
2. `copy .env.example .env`
3. `.\venv\Scripts\pip install -r requirements.txt`
4. `.\venv\Scripts\python manage.py migrate`
5. `.\start_xai_verify.bat`

Open http://127.0.0.1:8000

## Train deepfake detector

**Demo (one command — seed + train):**

```powershell
python manage.py train_demo_deepfake --dataset datasets
```

Or step by step:

```powershell
python manage.py seed_demo_dataset --dataset datasets
python manage.py train_deepfake --dataset datasets
```

**Your own images:**

```
datasets/real/   ← real photos
datasets/fake/   ← ChatGPT / AI images

python manage.py train_deepfake --dataset datasets
```

## Train fake news models

**TF-IDF (default in the web app, fast):**

```powershell
python manage.py train_fake_news
```

**BERT + RoBERTa (WELFake fine-tune):**

```powershell
python manage.py train_fake_news_transformers --architecture both
```

On CPU this uses 12,000 training articles by default (~1–3 hours). Full WELFake (~44k rows):

```powershell
python manage.py train_fake_news_transformers --architecture both --full --epochs 3
```

Requires `datasets/text/WELFake_Dataset.csv` and `torch`, `transformers`.

**Google Colab (GPU):** see [docs/COLAB_FAKE_NEWS_TRAINING.md](docs/COLAB_FAKE_NEWS_TRAINING.md) or open `notebooks/colab_train_fake_news.ipynb`.

## URLs

| URL | Page |
|-----|------|
| `/` | Dashboard |
| `/deepfake/` | DeepFake Detection |
| `/misinformation/` | Misinformation hub |
| `/misinformation/fake-news/` | Fake News |
| `/misinformation/recruitment-scam/` | Recruitment Scam |
| `/reports/history/` | Analysis History |
| `/about/` | About Research |

Legacy redirects: `/recruitment/` → recruitment scam, `/analysis/` → reports.

## Documentation

- [Folder structure](docs/FOLDER_STRUCTURE.md)
- [Installation guide](docs/INSTALLATION.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [API documentation](docs/API.md)
- [Research architecture](docs/ARCHITECTURE.md)
- [Model details (understanding guide)](docs/MODELS.md)
- [Dataset plan (real, screenshots, AI images)](docs/DATASET_GUIDE.md)
