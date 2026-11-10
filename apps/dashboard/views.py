from django.shortcuts import render


def home(request):
    return render(request, "dashboard/home.html")


def about(request):
    return render(request, "dashboard/about.html")
