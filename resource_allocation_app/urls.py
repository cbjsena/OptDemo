from django.urls import path
from . import views

app_name = 'resource_allocation_app' # 앱 네임스페이스

urlpatterns = [
    # General Introduction Page
    path('', views.resource_allocation_introduction_view, name='resource_allocation_introduction'),

    # Budget Allocation
    path('budget-allocation/introduction/', views.budget_allocation_introduction_view, name='budget_allocation_introduction'),
    # path('budget-allocation/data/', views.budget_allocation_data_view, name='budget_allocation_data'),
    path('budget-allocation/demo/', views.budget_allocation_demo_view, name='budget_allocation_demo'),

    # Financial Portfolio
    path('financial-portfolio/introduction/', views.financial_portfolio_introduction_view, name='financial_portfolio_introduction'),
    path('financial-portfolio/demo/', views.financial_portfolio_demo_view, name='financial_portfolio_demo'),

    # Data Center Capacity Planning
    path('data-center-capacity/introduction/', views.data_center_capacity_introduction_view,
         name='data_center_capacity_introduction'),
    path('data-center-capacity/demo/', views.data_center_capacity_demo_view, name='data_center_capacity_demo')

]