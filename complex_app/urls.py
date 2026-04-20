from django.urls import path

from . import views

app_name = 'complex_app'

urlpatterns = [
    path('', views.complex_app_introduction_view, name='complex_app_introduction'),
    path('palletizing/introduction/', views.palletizing_introduction_view, name='palletizing_introduction'),
    path('palletizing/demo/', views.palletizing_demo_view, name='palletizing_demo'),
]

