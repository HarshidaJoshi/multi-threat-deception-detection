# API documentation

XAI-Verify is primarily a **server-rendered Django application**. There is no separate REST API in the active stack. Endpoints are HTML form POST handlers.

## Web routes

| Method | URL | Name | Description |
|--------|-----|------|-------------|
| GET | `/` | `dashboard` | Dashboard home |
| GET | `/about/` | `about` | Dissertation about page |
| GET/POST | `/deepfake/` | `deepfake_analyze` | Upload image, run deepfake detection |
| GET | `/misinformation/` | `misinformation_index` | Misinformation module hub |
| GET/POST | `/misinformation/fake-news/` | `misinformation_fake_news` | Fake news text analysis |
| GET/POST | `/misinformation/recruitment-scam/` | `misinformation_recruitment_scam` | Recruitment scam analysis |
| GET | `/reports/history/` | `reports_history` | List past analyses |
| GET | `/reports/history/<id>/` | `reports_detail` | Analysis detail + XAI panel |
| GET | `/reports/history/<id>/report/` | `reports_download` | PDF report download |
| GET/POST | `/accounts/login/` | `login` | User login |
| POST | `/accounts/logout/` | `logout` | User logout |
| * | `/admin/` | — | Django admin |

## Legacy redirects

| Old URL | Redirects to |
|---------|--------------|
| `/recruitment/` | `/misinformation/recruitment-scam/` |
| `/analysis/` | `/reports/` |

## Analysis response shape (JSON in `result_json`)

All modules return a dict persisted in `AnalysisRecord.result_json`:

```json
{
  "module": "deepfake | misinformation | recruitment",
  "prediction": "string label",
  "confidence": 85.5,
  "fake_score": 72.0,
  "indicators": ["..."],
  "explainability": {
    "grad_cam": "base64 PNG (deepfake only)",
    "lime_tokens": [{"token": "...", "importance": 0.4, "direction": "fake"}],
    "shap_features": [{"feature": "...", "contribution_pct": 25.0}],
    "summary": "Key signals: ...",
    "reasons": ["..."]
  }
}
```

Recruitment adds: `fraud_risk_score`, `trust_score`, `website_status`, `employer_in_reference_list`.

## Management commands

```bash
python manage.py train_demo_deepfake --dataset datasets [--epochs 12] [--overwrite]
python manage.py seed_demo_dataset --dataset datasets [--count 60] [--overwrite]
python manage.py train_deepfake --dataset datasets [--epochs 12]
python manage.py migrate
python manage.py createsuperuser
```

## Python service API (internal)

```python
from utils.orchestration import analyze_deepfake, analyze_misinformation, analyze_recruitment

result = analyze_deepfake("/path/to/image.jpg")
result = analyze_misinformation("News text here...")
result = analyze_recruitment(job_text="...", company_name="...", website_url="...")
```

```python
from apps.explainability.services import generate_gradcam_heatmap, lime_text_explanation, shap_feature_importance
```

```python
from apps.reports.services import persist_analysis  # used by views
```
