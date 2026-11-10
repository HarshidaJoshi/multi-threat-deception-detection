from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.dashboard.urls")),
    path("accounts/", include("apps.accounts.urls")),
    path("deepfake/", include("apps.deepfake.urls")),
    path("misinformation/", include("apps.misinformation.urls")),
    path("reports/", include("apps.reports.urls")),
    # Legacy URL redirects (preserve bookmarks)
    path("analysis/", RedirectView.as_view(url="/reports/history/", permanent=False)),
    path("recruitment/", RedirectView.as_view(url="/misinformation/recruitment-scam/", permanent=False)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
