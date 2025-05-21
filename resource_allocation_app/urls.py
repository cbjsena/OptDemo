from django.urls import path
from . import views

app_name = 'resource_allocation_app' # 앱 네임스페이스

urlpatterns = [
    path('budget-allocation/introduction/', views.budget_allocation_introduction_view, name='budget_allocation_introduction'),
    # path('budget-allocation/data/', views.budget_allocation_data_view, name='budget_allocation_data'),
    path('budget-allocation/demo/', views.budget_allocation_demo_view, name='budget_allocation_demo'),
]