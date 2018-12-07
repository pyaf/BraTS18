# from django_cron import CronJobBase, Schedule
import os
import requests
from .models import Scan
from django.conf import settings

DL_SERVER = settings.DL_SERVER
BASE_DIR = settings.BASE_DIR


def mkdir(folder):
    print(os.path.exists(folder))
    if not os.path.exists(folder):
        os.makedirs(folder)


def my_scheduled_job():
    scans = Scan.objects.filter(seg_file="")  # '' is imp
    print("*" * 20)
    print("Cron:", len(scans))
    for scan in scans:
        scan_id = scan.scan_id
        patient_id = scan.patient.id
        seg_file_folder = os.path.join(BASE_DIR, "files/segmentations", str(patient_id))
        print(seg_file_folder)
        mkdir(seg_file_folder)
        seg_file_path = os.path.join(seg_file_folder, "%s.nii.gz" % scan_id)
        url = DL_SERVER + "scanStatus"
        data = {"patient_id": patient_id, "scan_id": scan_id}
        r = requests.post(url, params=data)
        if r.status_code == 200:
            with open(seg_file_path, "wb") as f:
                f.write(r.content)
                scan.seg_file = "segmentations/%s/%s.nii.gz" % (
                    patient_id,
                    scan_id,
                )  # 1
                scan.save()
        elif r.status_code == 404:
            print("Segmentation not ready..")


# my_scheduled_job()

# ## FOOTNOTES ###

# 1: django model path saves relative to 'files'

# > crontab -e

# grep CRON /var/log/syslog  #cronjob log, doesn't contain the outputs
# sudo tail -f /var/mail/ags  ## check outputs of cronjob

# python manage.py crontab add/remove/show
