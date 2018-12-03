from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path("", views.index),
    url(r"^patients/(?P<pid>[-\w]+)/", views.patientsView),
    url(r"^analyze/(?P<scan_id>[-\w]+)/", views.analyze),
    path("login/", views.loginView),
    path("logout/", views.logoutView),
]
