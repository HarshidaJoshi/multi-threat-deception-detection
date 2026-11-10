from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    path("", views.index_view, name="misinformation_index"),
    path("fake-news/", views.fake_news_view, name="misinformation_fake_news"),
    path("recruitment-scam/", views.recruitment_scam_view, name="misinformation_recruitment_scam"),
    # Backward-compatible redirects
    path(
        "analyze/",
        RedirectView.as_view(pattern_name="misinformation_fake_news", permanent=False),
        name="misinformation_analyze",
    ),
]
