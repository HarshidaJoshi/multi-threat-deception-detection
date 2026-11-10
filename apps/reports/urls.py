from django.urls import path

from . import views

urlpatterns = [
    path("history/", views.history_list, name="reports_history"),
    path("history/<int:record_id>/", views.history_detail, name="reports_detail"),
    path("history/<int:record_id>/report/", views.download_report, name="reports_download"),
]
