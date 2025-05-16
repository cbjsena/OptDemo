from django.urls import path
from . import views
app_name = 'matching_app'  # URL 네임스페이스 설정
urlpatterns = [
    path('data-generation/', views.matching_data_generation_view, name='data_generation'),
    path('small-scale-demo/', views.matching_small_scale_demo_view, name='small_scale_demo'),
    path('large-scale-demo/', views.matching_large_scale_demo_view, name='large_scale_demo'),
    # 필요에 따라 다른 matching_app 관련 URL 추가
]