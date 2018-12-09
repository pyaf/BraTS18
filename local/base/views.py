from django.shortcuts import render, redirect, HttpResponse
from django.http import Http404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from base.models import Patient, Scan
from subprocess import call
from threading import Thread, active_count
from django.conf import settings
import pdb
import sys
import os
from io import BytesIO
from glob import glob
import json
import zipfile
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile

# sys.path.append(os.path.abspath('BraTS'))
# from BraTS import testBraTS

DL_SERVER = settings.DL_SERVER


def extract_zip(zipscanpath):
    scan_id = zipscanpath.split("/")[-1].split(".")[0]
    patient_id = zipscanpath.split("/")[-2]
    extraction_path = os.path.join("files/scans", patient_id, scan_id)
    with zipfile.ZipFile(zipscanpath, "r") as zip_ref:
        zip_ref.extractall(extraction_path)  # footnote 1


@login_required(login_url="/login")
def index(request):
    patients = Patient.objects.filter(doctor=request.user)
    context = {"patients": patients}
    return render(request, "patient_list.html", context)


def postScan(scan):
    """Post scan to DL_server"""
    scanpath = scan.file.path
    patient_id = scan.patient.id
    data = {"patient_id": patient_id}
    url = DL_SERVER + "postScans"
    with open(scanpath, "rb") as scans:
        r = requests.post(url, files={"scans": scans}, params=data)
        print(r.status_code)
        response = json.loads(r.text)
        if response["success"]:
            print("Scan post Successful!")
        else:
            print("Error in posting scans")


def itkSnap(request, patient_id, scan_id):
    if request.user.is_authenticated:
        patient = Patient.objects.get(id=patient_id)
        scan = Scan.objects.get(patient=patient, scan_id=scan_id)
        patient_id = scan.patient.id
        scan_id = scan.scan_id
        file = glob("files/scans/%s/%s/*t1ce.nii.gz" % (patient_id, scan_id))[0]
        t1file = glob("files/scans/%s/%s/*t1.nii.gz" % (patient_id, scan_id))[0]
        t2file = glob("files/scans/%s/%s/*t2.nii.gz" % (patient_id, scan_id))[0]
        flairfile = glob("files/scans/%s/%s/*flair.nii.gz" % (patient_id, scan_id))[0]

        if scan.seg_file:
            seg_file = scan.seg_file.path
            cmd = "itksnap -g %s -s %s -o %s %s %s -l %s" % (
                file,
                seg_file,
                t1file,
                t2file,
                flairfile,
                "files/labels.txt",
            )
        else:
            cmd = "itksnap -g %s" % file
        out = call(cmd.split())
        return HttpResponse("Done")
    else:
        raise Http404()


@login_required(login_url="/login")
def patientsView(request, pid):
    try:
        patient = Patient.objects.get(id=pid)
        # scan = Scan.objects.filter(patient=patient)[0]
        # Thread(target=postScan, args=(scan,)).start()

    except Exception as e:
        raise Http404(e)
    if request.method == "POST" and request.FILES["scans"]:
        patient_id = patient.id
        files = request.FILES.getlist("scans")
        bytes_obj = BytesIO()
        zf = zipfile.ZipFile(bytes_obj, "w")

        for file in files:
            fpath = file.temporary_file_path()  # django stores them at /tmp
            fname = file.name
            zf.write(fpath, fname)
        zf.close()
        tempzipfile = InMemoryUploadedFile(
            file=bytes_obj,
            field_name=None,
            name="lol",
            content_type="application/zip",
            size=bytes_obj.__sizeof__(),
            charset=None,
        )
        description = request.POST.get("description", None)
        scan = Scan.objects.create(
            patient=patient, file=tempzipfile, description=description
        )
        Thread(target=postScan, args=(scan,)).start()
        Thread(target=extract_zip, args=(scan.file.path,)).start()
        return redirect("/patients/%s" % pid)
    patient = Patient.objects.get(id=pid)
    scans = Scan.objects.filter(patient=patient)
    context = {"patient": patient, "scans": scans}
    return render(request, "patient_profile.html", context)


def loginView(request):
    if request.user.is_authenticated:
        return redirect("/")
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.add_message(request, messages.INFO, "Login Successful!")
            return redirect("/")
        else:
            messages.add_message(request, messages.INFO, "Invalid Login Credentials!")
    return render(request, "login.html")


def editPatientProfile(request):
    try:
        if request.method == "POST":
            patient_id = request.POST["patient_id"]
            patient = Patient.objects.get(id=patient_id)
            patient.name = request.POST["name"]
            patient.email = request.POST["email"]
            patient.age = request.POST["age"]
            patient.gender = request.POST["gender"]
            patient.address = request.POST["address"]
            patient.mobile_number = request.POST["mobile_number"]
            patient.description = request.POST["description"]
            patient.save()
            print("Updated profile")
            return redirect("/patients/%s" % patient_id)
    except Exception as e:
        print(e)
        return redirect("/")


def newPatient(request):
    if request.method == "POST":
        print(request.POST)
        print(request.FILES)
        # pdb.set_trace()
        doctor = request.user
        name = request.POST['name']
        gender = request.POST['gender']
        age = request.POST['age']
        mobile_number = request.POST['mobile_number']
        description = request.POST['description']
        picture = request.FILES['picture']
        email = request.POST['email']
        patient = Patient.objects.create(
            doctor=doctor,
            name=name,
            gender=gender,
            age=age,
            mobile_number=mobile_number,
            description=description,
            picture=picture,
            email=email
            )
        print(patient)
        return redirect('/')
    return render(request, 'new_patient.html')


@login_required(login_url="/login")
def logoutView(request):
    logout(request)
    messages.add_message(request, messages.INFO, "Logout Successful!")
    return redirect("/")
