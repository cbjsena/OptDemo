from django.urls import path
from . import views

app_name = 'core' # URL 네임스페이스 설정

urlpatterns = [
    path('', views.home_view, name='home'),
    path('core/ready/', views.ready_view, name='ready'),
]