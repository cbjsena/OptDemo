from django.urls import path
from . import views

app_name = 'resource_allocation_app' # 앱 네임스페이스

urlpatterns = [
    # General Introduction Page
    path('', views.resource_allocation_introduction_view, name='resource_allocation_introduction'),

    # 1. Budget Allocation
    path('budget-allocation/introduction/', views.budget_allocation_introduction_view, name='budget_allocation_introduction'),
    path('budget-allocation/demo/', views.budget_allocation_demo_view, name='budget_allocation_demo'),

    # 2.Data Center Capacity Planning
    path('data-center-capacity/introduction/', views.data_center_capacity_introduction_view,
         name='data_center_capacity_introduction'),
    path('data-center-capacity/demo/', views.data_center_capacity_demo_view, name='data_center_capacity_demo'),

    # 3. Nurse Rostering Problem
    path('nurse-rostering/introduction/', views.nurse_rostering_introduction_view, name='nurse_rostering_introduction'),
    path('nurse-rostering/demo/', views.nurse_rostering_demo_view, name='nurse_rostering_demo'),
    path('nurse-rostering/advanced-demo/', views.nurse_rostering_advanced_demo_view, name='nurse_rostering_advanced_demo'),

    path('fleet-cascading/introduction0/', views.fleet_cascading_introduction0_view, name='fleet_cascading_introduction0'),
    path('fleet-cascading/demo0/', views.fleet_cascading_demo0_view, name='fleet_cascading_demo0'),
    path('fleet-cascading/introduction/', views.fleet_cascading_introduction_view, name='fleet_cascading_introduction'),
    path('fleet-cascading/demo/', views.fleet_cascading_demo_view, name='fleet_cascading_demo'),

    # Financial Portfolio
    path('financial-portfolio/introduction/', views.financial_portfolio_introduction_view,
         name='financial_portfolio_introduction'),
    path('financial-portfolio/demo/', views.financial_portfolio_demo_view, name='financial_portfolio_demo'),

]