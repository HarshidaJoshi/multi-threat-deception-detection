# Deployment guide

## Production checklist

1. Set `DJANGO_DEBUG=0` in `.env`
2. Set a strong `DJANGO_SECRET_KEY`
3. Set `DJANGO_ALLOWED_HOSTS` to your domain/IP
4. Set `XAI_DEMO_MODE=0`
5. Configure MySQL with a dedicated user (not root)
6. Run `python manage.py collectstatic`
7. Serve with **Gunicorn** + **Nginx** (Linux) or **Waitress** (Windows)

## Windows (Waitress)

```powershell
pip install waitress
.\venv\Scripts\python -c "from waitress import serve; from xai_verify.wsgi import application; serve(application, host='0.0.0.0', port=8000)"
```

## Linux (Gunicorn example)

```bash
pip install gunicorn
gunicorn xai_verify.wsgi:application --bind 0.0.0.0:8000 --workers 2
```

Use Nginx as reverse proxy for static/media files.

## Environment variables

| Variable | Production value |
|----------|------------------|
| `DJANGO_SECRET_KEY` | Random 50+ char string |
| `DJANGO_DEBUG` | `0` |
| `DJANGO_ALLOWED_HOSTS` | `yourdomain.com,www.yourdomain.com` |
| `DB_NAME` | `xai_verify` |
| `DB_USER` | Dedicated DB user |
| `DB_PASSWORD` | Strong password |
| `DEEPFAKE_MODELS_DIR` | Absolute path to `trained_models/` |

## Media & models

- Mount `media/` and `trained_models/` on persistent storage
- Back up MySQL regularly (`analysis_analysisrecord` table)
- Pre-train deepfake model before deploy to avoid CPU training in production

## Security notes

- Keep `.env` out of version control
- Limit upload size (configured in `xai_verify/settings/base.py`)
- Use HTTPS in production
- Restrict `/admin/` by IP or strong credentials
