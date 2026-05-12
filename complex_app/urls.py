from django.urls import path

from .views import (palletizing_introduction_view, palletizing_demo_view,
                    complex_app_introduction_view, lsnd_introduction_view,
                    lsnd_advanced_model_view, lsnd_benchmark_data_view,
                    vessel_deployment_introduction_view, vessel_deployment_demo_view,
                    vessel_deployment_demo2_view, vessel_deployment_advanced_model_view,
                    vessel_deployment_demo3_view)

app_name = 'complex_app'

urlpatterns = [
    path('', complex_app_introduction_view, name='complex_app_introduction'),
    path('palletizing/introduction/', palletizing_introduction_view, name='palletizing_introduction'),
    path('palletizing/demo/', palletizing_demo_view, name='palletizing_demo'),
    path('lsnd/introduction/', lsnd_introduction_view, name='lsnd_introduction'),
    path('lsnd/advanced-model/', lsnd_advanced_model_view, name='lsnd_advanced_model'),
    path('lsnd/benchmark-data/', lsnd_benchmark_data_view, name='lsnd_benchmark_data'),
    path('vessel-deployment/introduction/', vessel_deployment_introduction_view, name='vessel_deployment_introduction'),
    path('vessel-deployment/demo/', vessel_deployment_demo_view, name='vessel_deployment_demo'),
    path('vessel-deployment/demo2/', vessel_deployment_demo2_view, name='vessel_deployment_demo2'),
    path('vessel-deployment/advanced-model/', vessel_deployment_advanced_model_view, name='vessel_deployment_advanced_model'),
    path('vessel-deployment/demo3/', vessel_deployment_demo3_view, name='vessel_deployment_demo3'),
]
