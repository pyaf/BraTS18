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
import sys
import os
sys.path.append(os.path.abspath('BraTS'))
from BraTS import testBraTS

@login_required(login_url="/login")
def index(request):
    patients = Patient.objects.filter(doctor=request.user)
    context = {"patients": patients}
    return render(request, "patient_list.html", context)


def generate_scan(scan):
    '''pass scan file to the model, generate segmentation, update scan object'''
    pass


def analyze(request, scan_id):
    if request.user.is_authenticated:
        scan = Scan.objects.get(scan_id=scan_id)
        file = scan.file.path
        if scan.seg_file:
            seg_file = scan.seg_file.path
            cmd = "itksnap -g %s -s %s" % (file, seg_file)
        else:
            cmd = "itksnap -g %s" % file
        out = call(cmd.split())
        return HttpResponse('Done')
    else:
        raise Http404()


@login_required(login_url="/login")
def patientsView(request, pid):
    try:
        patient = Patient.objects.get(id=pid)
    except Exception as e:
        raise Http404(e)
    if request.method == "POST" and request.FILES["scan"]:
        file = request.FILES["scan"]
        description = request.POST.get('description', None)
        scan = Scan.objects.create(patient=patient, file=file, description=description)
        Thread(target=generate_scan, args=(scan)).start()
        return redirect('/patients/%s' % pid)
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


@login_required(login_url="/login")
def logoutView(request):
    logout(request)
    messages.add_message(request, messages.INFO, "Logout Successful!")
    return redirect("/")
