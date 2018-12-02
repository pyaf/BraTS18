from django.shortcuts import render, redirect, HttpResponse
from django.http import Http404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from base.models import Patient, Scan


@login_required(login_url="/")
def index(request):
    patients = Patient.objects.filter(doctor=request.user)
    context = {"patients": patients}
    return render(request, "index.html", context)


@login_required(login_url="/")
def patientsView(request, pid):
    try:
        patient = Patient.objects.get(id=pid)
    except Exception as e:
        raise Http404(e)
    if request.method == "POST" and request.FILES["scan"]:
        file = request.FILES["scan"]
        description = request.POST.get('description', None)
        Scan.objects.create(patient=patient, file=file, description=description)
        return redirect('/patients/%s' % pid)
    patient = Patient.objects.get(id=pid)
    scans = Scan.objects.filter(patient=patient)
    context = {"patient": patient, "scans": scans}
    return render(request, "patient.html", context)


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


@login_required(login_url="/")
def logoutView(request):
    logout(request)
    messages.add_message(request, messages.INFO, "Logout Successful!")
    return redirect("/")
