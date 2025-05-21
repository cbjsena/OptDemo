from django.shortcuts import render
from django.urls import reverse

def home_view(request):
    context = {
        'active_model': None,
        'active_submenu': 'home', # 또는 None
        'home_url': reverse('core:home') # 홈페이지 URL을 context에 추가
    }
    return render(request, 'core/home.html', context)


def ready_view(request):
    context = {
        'active_model': None,
        'active_submenu': 'ready',  # 또는 None
        'home_url': reverse('core:ready')  # 홈페이지 URL을 context에 추가
    }
    return render(request, 'core/ready.html', context)