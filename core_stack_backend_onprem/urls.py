"""
URL configuration for core_stack_backend_onprem project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path

from .api import generate_farm_boundary, generate_wells_layer, generate_ponds_layer

urlpatterns = [
    path("admin/", admin.site.urls),
    path("compute/farm-boundary/", generate_farm_boundary),
    path("compute/ponds/", generate_ponds_layer),
    path("compute/wells/", generate_wells_layer),
]
