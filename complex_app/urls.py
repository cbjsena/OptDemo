from django.urls import path

from .views import (palletizing_introduction_view, palletizing_demo_view,
                    complex_app_introduction_view, lsnd_introduction_view,
                    lsnd_advanced_model_view, lsnd_benchmark_data_view)

app_name = 'complex_app'

urlpatterns = [
    path('', complex_app_introduction_view, name='complex_app_introduction'),
    path('palletizing/introduction/', palletizing_introduction_view, name='palletizing_introduction'),
    path('palletizing/demo/', palletizing_demo_view, name='palletizing_demo'),
    path('lsnd/introduction/', lsnd_introduction_view, name='lsnd_introduction'),
    path('lsnd/advanced-model/', lsnd_advanced_model_view, name='lsnd_advanced_model'),
    path('lsnd/benchmark-data/', lsnd_benchmark_data_view, name='lsnd_benchmark_data'),
]
