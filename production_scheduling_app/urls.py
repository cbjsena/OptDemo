from django.urls import path
from . import views

app_name = 'production_scheduling_app' # 앱 네임스페이스

urlpatterns = [
    # General Introduction Page
    path('', views.production_scheduling_introduction_view, name='production_scheduling_introduction'),

    # 1. Lot Sizing Problem
    path('lot-sizing/introduction/', views.lot_sizing_introduction_view, name='lot_sizing_introduction'),
    path('lot-sizing/demo/', views.lot_sizing_demo_view, name='lot_sizing_demo'),

    # 2. Single Machine Scheduling
    path('single-machine/introduction/', views.single_machine_introduction_view, name='single_machine_introduction'),
    path('single-machine/demo/', views.single_machine_demo_view, name='single_machine_demo'),
    path('single-machine/advanced/', views.single_machine_advanced_view, name='single_machine_advanced'),


    # 3. Flow Shop Scheduling
    path('flow-shop/introduction/', views.flow_shop_introduction_view, name='flow_shop_introduction'),
    path('flow-shop/demo/', views.flow_shop_demo_view, name='flow_shop_demo'),

    # 4. Job Shop Scheduling Problem
    path('job-shop/introduction/', views.job_shop_introduction_view, name='job_shop_introduction'),
    path('job-shop/demo/', views.job_shop_demo_view, name='job_shop_demo'),

    # 5. RCPSP
    path('rcpsp/introduction/', views.rcpsp_introduction_view, name='rcpsp_introduction'),
    path('rcpsp/demo/', views.rcpsp_demo_view, name='rcpsp_demo'),
]