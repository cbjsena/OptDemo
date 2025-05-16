from django.urls import path
from . import views

app_name = 'core' # URL 네임스페이스 설정

urlpatterns = [
    path('', views.home_view, name='home'),
    # 필요에 따라 다른 core 관련 URL 추가
]