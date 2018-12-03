import os
from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
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
    return "scans/%d/%s.nii.gz" % (instance.patient.id, instance.scan_id)


def get_seg_path(instance, filename):
    return "segmentations/%d/%s.nii.gz" % (instance.patient.id, instance.scan_id)


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
    scan_id = models.CharField(max_length=200, null=True)
    description = models.TextField(null=True, blank=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    file = models.FileField(blank=True, upload_to=get_scan_path)
    seg_file = models.FileField(null=True, upload_to=get_seg_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s-%s' % (self.patient.name, self.scan_id)

    def save(self, *args, **kwargs):
        scans = Scan.objects.filter(patient=self.patient)
        self.scan_id = len(scans) + 1
        print(self.file)
        self.description += "\nFilename: " + str(self.file)
        super(Scan, self).save(*args, **kwargs)



@receiver(models.signals.post_delete, sender=Scan)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding `Scan` object is deleted.
    """
    if instance.file:
        if os.path.isfile(instance.file.path):
            os.remove(instance.file.path)
    if instance.seg_file:
        if os.path.isfile(instance.seg_file.path):
            os.remove(instance.seg_file.path)