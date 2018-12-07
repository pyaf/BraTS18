from django.contrib import admin
from .models import Patient, Scan

# Register your models here.
admin.site.register(Patient)
admin.site.register(Scan)
