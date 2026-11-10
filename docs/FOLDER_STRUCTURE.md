# Folder structure

```
deepfake_detection_project/
│
├── manage.py                 # Django entry point
├── requirements.txt
├── setup_xampp.sql
├── start_xai_verify.bat
├── run_server.bat
├── .env / .env.example
│
├── xai_verify/               # Django project package (settings = configs)
│   ├── settings/base.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── apps/                     # Django applications
│   ├── dashboard/            # Home, about, site context
│   ├── deepfake/             # Image upload & analysis views
│   ├── misinformation/       # Fake news + recruitment scam (sub-module)
│   ├── explainability/       # Grad-CAM, LIME, SHAP service exports
│   ├── reports/              # AnalysisRecord model, history, PDF
│   └── accounts/             # Authentication views
│
├── utils/                    # ML & report utilities (framework-agnostic)
│   ├── deepfake_classifier.py
│   ├── deepfake_models.py
│   ├── text_models.py
│   ├── recruitment.py
│   ├── ml_utils.py
│   ├── orchestration.py
│   ├── explainability.py
│   └── reports.py
│
├── templates/                # Django HTML templates
├── static/css/               # xai_verify.css
├── media/                    # User uploads (runtime)
├── datasets/                 # Training data
├── trained_models/           # Saved model weights
├── generated_reports/        # Optional PDF output directory
├── docs/                     # Project documentation
└── legacy/                   # Notes about removed prototypes
```

## App responsibilities

| App | Role |
|-----|------|
| `dashboard` | Landing page, dissertation about page, template context |
| `deepfake` | Image form, calls `utils.orchestration.analyze_deepfake` |
| `misinformation` | Fake news & recruitment scam forms |
| `explainability` | Documents and re-exports XAI helpers |
| `reports` | Persists `AnalysisRecord`, history list/detail, PDF |
| `accounts` | Django auth login/logout |

## Database

Single custom table: `analysis_analysisrecord` (preserved from earlier migrations via `db_table` in `apps.reports.models`).

Module values: `deepfake`, `misinformation`, `recruitment` (recruitment stored as misinformation sub-category).
