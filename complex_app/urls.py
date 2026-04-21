from django.urls import path

from .views import palletizing_introduction_view, palletizing_demo_view, complex_app_introduction_view

app_name = 'complex_app'

urlpatterns = [
    path('', complex_app_introduction_view, name='complex_app_introduction'),
    path('palletizing/introduction/', palletizing_introduction_view, name='palletizing_introduction'),
    path('palletizing/demo/', palletizing_demo_view, name='palletizing_demo'),
]
