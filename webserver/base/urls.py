from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path("", views.index),
    url("patients/(?P<pid>[-\w]+)/", views.patientsView),
    path("login/", views.loginView),
    path("logout/", views.logoutView),
]
