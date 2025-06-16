from django.shortcuts import render
import logging
import random
import json

from .utils import data_utils
from common_utils.run_production_opt import *
from common_utils.default_data import (
    preset_budjet_items,
    preset_datacenter_servers,
    preset_datacenter_services, preset_num_periods
)
from .utils.data_utils import create_lot_sizing_json_data, save_production_json_data

logger = logging.getLogger(__name__)  # settings.py에 정의된 'resource_allocation_app' 로거 사용


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


def lot_sizing_demo_view(request):
    """
    Lot Sizing Problem 데모 뷰.
    """
    form_data = {}

    if request.method == 'GET':
        submitted_num_periods = int(request.GET.get('num_periods_to_show', preset_num_periods))
        submitted_num_periods = max(3, min(12, submitted_num_periods))

        # GET 요청 시 랜덤 기본값으로 form_data 초기화
        for t in range(submitted_num_periods):
            form_data[f'demand_{t}'] = request.GET.get(f'demand_{t}', str(random.randint(50, 150)))
            form_data[f'setup_cost_{t}'] = request.GET.get(f'setup_cost_{t}', str(random.randint(200, 500)))
            form_data[f'prod_cost_{t}'] = request.GET.get(f'prod_cost_{t}', str(random.randint(5, 15)))
            form_data[f'holding_cost_{t}'] = request.GET.get(f'holding_cost_{t}', str(random.randint(1, 5)))
            form_data[f'capacity_{t}'] = request.GET.get(f'capacity_{t}', str(random.randint(150, 300)))

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_periods = int(form_data.get('num_periods', preset_num_periods))

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'lot_sizing_demo',
        'form_data': form_data,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_periods_options': range(3, 13),  # 3~12 기간
        'submitted_num_periods': submitted_num_periods,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info("Lot Sizing Demo POST request processing.")
        try:
            # 1. 데이터 생성 및 검증
            input_data = create_lot_sizing_json_data(form_data, submitted_num_periods)

            # 2. 파일 저장
            saved_filename, save_error = save_production_json_data (input_data)
            if save_error:
                context['error_message'] = save_error
            elif saved_filename:
                context['info_message'] = f"입력 데이터가 '{saved_filename}'으로 서버에 저장되었습니다."

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time_ms = run_lot_sizing_optimizer(input_data)
            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적 생산 계획 수립 완료! 최소 총 비용: {results_data['total_cost']:.2f}"

                # 차트용 데이터 준비
                chart_data = {
                    'labels': [f'기간 {p["period"]}' for p in results_data['schedule']],
                    'demands': [p['demand'] for p in results_data['schedule']],
                    'productions': [p['production_amount'] for p in results_data['schedule']],
                    'inventories': [p['inventory_level'] for p in results_data['schedule']],
                }
                context['plot_data'] = json.dumps(chart_data)

            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'production_scheduling_app/lot_sizing_demo.html', context)


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
