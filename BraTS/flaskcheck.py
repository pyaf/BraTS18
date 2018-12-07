import requests
import json

# filepath = 'test/1.zip'
# url = 'http://localhost:6006/postScans'
# patient_id = 24
# data = {'patient_id': patient_id}
# with open(filepath, 'rb') as scans:
#     r = requests.post(url, files={'scans': scans}, params=data)
#     print(r.status_code)
#     print(r.text)

# import os
# import zipfile
# import pdb
# from glob import glob
# zipscanpath = 'files/scans/24/1.zip'
# scan_id = zipscanpath.split('/')[-1].split('.')[0]
# patient_id = zipscanpath.split('/')[-2]


# archive = zipfile.ZipFile(zipscanpath, 'r')
# # zipfolder = archive.namelist[0]
# extraction_path = os.path.join('files/scans', patient_id, scan_id)
# # with zipfile.ZipFile(zipscanpath, 'r') as zip_ref:
# #     zip_ref.extractall(extraction_path)


# files = glob(extraction_path + '/*/*.nii*')

# pdb.set_trace()


import requests
import json, pdb, shutil

url = "http://localhost:6006/scanStatus"
patient_id = 24
scan_id = 1
data = {"patient_id": patient_id, "scan_id": scan_id}

r = requests.post(url, params=data)
if r.status_code == 200:
    with open("lol.nii.gz", "wb") as f:
        f.write(r.content)
elif r.status_code == 404:
    print("Segmentation not ready..")
