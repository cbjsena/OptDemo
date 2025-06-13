from django.shortcuts import render
import logging

logger = logging.getLogger(__name__)

def production_scheduling_introduction_view(request):
    """General introduction page for the Production & Scheduling category."""
    context = {
        'active_model': 'Production & Scheduling',
        # 이 페이지는 특정 소메뉴에 속하지 않으므로 active_submenu는 비워둠
        'active_submenu': 'main_introduction'
    }
    logger.debug("Rendering general Production & Scheduling introduction page.")
    return render(request, 'production_scheduling_app/production_scheduling_introduction.html', context)

def lot_sizing_introduction_view(request):
    """Lot Sizing Problem Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'lot_sizing',
        'active_submenu': 'lot_sizing_introduction'
    }
    logger.debug("Rendering Lot Sizing introduction page.")
    # 실제 템플릿 파일을 생성해야 합니다.
    # return render(request, 'production_scheduling_app/lot_sizing_introduction.html', context)
    return render(request, 'production_scheduling_app/lot_sizing_introduction.html', context) # 임시 페이지

def single_machine_introduction_view(request):
    """Single Machine Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'single_machine',
        'active_submenu': 'single_machine_introduction'
    }
    logger.debug("Rendering Single Machine Scheduling introduction page.")
    return render(request, 'production_scheduling_app/single_machine_introduction.html', context)

def flow_shop_introduction_view(request):
    """Flow Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'flow_shop',
        'active_submenu': 'flow_shop_introduction'
    }
    logger.debug("Rendering Flow Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/flow_shop_introduction.html', context)

def job_shop_introduction_view(request):
    """Job Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'job_shop',
        'active_submenu': 'job_shop_introduction'
    }
    logger.debug("Rendering Job Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/job_shop_introduction.html', context)

def rcpsp_introduction_view(request):
    """RCPSP Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'rcpsp',
        'active_submenu': 'rcpsp_introduction'
    }
    logger.debug("Rendering RCPSP introduction page.")
    return render(request, 'production_scheduling_app/rcpsp_introduction.html', context)

# 참고: 실제 소개 페이지 템플릿이 준비될 때까지 사용할 임시 템플릿
# `production_scheduling_app/templates/production_scheduling_app/under_construction.html` 파일을 생성하고
# "{% extends 'core/base.html' %} <h1>페이지 준비 중</h1>" 과 같이 간단히 작성해두면 좋습니다.