from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path("", views.index),
    url(r"^patients/(?P<pid>[-\w]+)/", views.patientsView),
    path("newPatient/", views.newPatient),
    url(r"^analyze/(?P<patient_id>[-\w]+)/(?P<scan_id>[-\w]+)/", views.itkSnap),
    path("editPatientProfile/", views.editPatientProfile),
    path("login/", views.loginView),
    path("logout/", views.logoutView),
]
