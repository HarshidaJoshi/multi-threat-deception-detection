from django.conf import settings


def site_settings(request):
    return {
        "demo_mode": settings.DEMO_MODE,
        "app_name": "XAI-Verify",
        "research_title": settings.RESEARCH_TITLE,
    }
