import os
from django.db import models
from django.contrib.auth.models import User

# Create your models here.

GENDER_CHOICES = (
    ("Male", "Male"),
    ("Female", "Female"),
    ("Other", "Other"),
)


def get_file_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = "%s.%s" % (instance.id, ext)
    return os.path.join('profile_pictures', filename)


def get_scan_path(instance, filename):
    return "scans/%d/%s" % (instance.patient.id, filename)


class Patient(models.Model):
    doctor = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.IntegerField()
    mobile_number = models.BigIntegerField()
    address = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    picture = models.FileField(null=True, blank=True, upload_to=get_file_path, default="files/default.png")
    email = models.EmailField()

    def __str__(self):
        return self.name


class Scan(models.Model):
    description = models.TextField(null=True, blank=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    file = models.FileField(blank=True, upload_to=get_scan_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
