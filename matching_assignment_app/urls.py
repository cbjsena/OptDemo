from django.urls import path
from . import views

app_name = 'matching_assignment_app' # 앱 네임스페이스

urlpatterns = [
    # LCD cf-tft Matching (기존 matching_app에서 가져오거나 여기에 새로 정의)
    path('lcd-cf-tft/introduction/', views.lcd_cf_tft_introduction_view, name='lcd_cf_tft_introduction'),
    path('lcd-cf-tft/data-generation/', views.lcd_cf_tft_data_generation_view, name='lcd_cf_tft_data_generation'),
    path('lcd-cf-tft/small-scale-demo/', views.lcd_cf_tft_small_scale_demo_view, name='lcd_cf_tft_small_scale_demo'),
    path('lcd-cf-tft/large-scale-demo/', views.lcd_cf_tft_large_scale_demo_view, name='lcd_cf_tft_large_scale_demo'),

    # 작업 배정 문제 (Assignment Problem)
    path('assignment-problem/introduction/', views.assignment_problem_introduction_view, name='assignment_problem_introduction'),
    # ... 데이터 생성, 데모 URL ...

    # 안정적인 매칭 문제 (Stable Matching Problem)
    path('stable-matching/introduction/', views.stable_matching_introduction_view, name='stable_matching_introduction'),
    # ... 데이터 생성, 데모 URL ...

    # 자원-기술 매칭 최적화 (Resource-Skill Matching)
    path('resource-skill-matching/introduction/', views.resource_skill_matching_introduction_view, name='resource_skill_matching_introduction'),
    # ... 데이터 생성, 데모 URL ...
]