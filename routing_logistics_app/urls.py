from django.urls import path
from . import views

app_name = 'routing_logistics_app'

urlpatterns = [
    # Vehicle Routing Problem (VRP)
    path('vrp/introduction/', views.vrp_introduction_view, name='vrp_introduction'),
    path('vrp/advanced/', views.vrp_advanced, name='vrp_advanced'),
    path('vrp/demo/', views.vrp_demo_view, name='vrp_demo'),

    # Capacitated VRP (CVRP) - 예시
    path('cvrp/introduction/', views.cvrp_introduction_view, name='cvrp_introduction'),
    path('cvrp/demo/', views.cvrp_demo_view, name='cvrp_demo'),

    # VRP with Time Windows (VRPTW) - 예시
    path('pdp/introduction/', views.pdp_introduction_view, name='pdp_introduction'),
    path('pdp/demo/', views.pdp_demo_view, name='pdp_demo'),
]