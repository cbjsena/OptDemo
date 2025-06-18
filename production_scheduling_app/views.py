from django.shortcuts import render
import json

from common_utils.run_production_opt import *
from common_utils.default_data import (
    preset_lot_sizing_num_periods,
    preset_single_machine_objective,
    preset_single_machine_objective_choice,
    preset_single_machine_num_jobs,
    preset_single_machine_data,
)
from common_utils.data_utils_production import *

logger = logging.getLogger(__name__)


def production_scheduling_introduction_view(request):
    """General introduction page for the Production & Scheduling category."""
    context = {
        'active_model': 'Production & Scheduling',
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
    return render(request, 'production_scheduling_app/lot_sizing_introduction.html', context)  # 임시 페이지


def lot_sizing_demo_view(request):
    """
    Lot Sizing Problem 데모 뷰.
    """
    form_data = {}

    if request.method == 'GET':
        submitted_num_periods = int(request.GET.get('num_periods_to_show', preset_lot_sizing_num_periods))
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
        submitted_num_periods = int(form_data.get('num_periods', preset_lot_sizing_num_periods))

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
            saved_filename, save_error = save_production_json_data(input_data)
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


def single_machine_demo_view(request):
    jobs_list = []  # 템플릿에 전달할 작업 데이터 리스트
    form_data_for_post = {}  # POST 데이터 처리를 위한 딕셔너리
    if request.method == 'GET':
        submitted_num_jobs = int(request.GET.get('num_jobs_to_show', preset_single_machine_num_jobs))
        submitted_num_jobs = max(2, min(8, submitted_num_jobs))  # 2~8개 작업

        for i in range(submitted_num_jobs):
            preset = preset_single_machine_data[i]
            jobs_list.append({
                'id': request.GET.get(f'job_{i}_id', preset['id']),
                'processing_time': request.GET.get(f'job_{i}_processing_time', preset['processing_time']),
                'due_date': request.GET.get(f'job_{i}_due_date', preset['due_date']),
            })

    elif request.method == 'POST':
        form_data_for_post = request.POST.copy()
        submitted_num_jobs = int(form_data_for_post.get('num_jobs', preset_single_machine_num_jobs))
        # POST 요청 시, 제출된 값으로 jobs_list를 채움 (입력값 유지)
        for i in range(submitted_num_jobs):
            jobs_list.append({
                'id': form_data_for_post.get(f'job_{i}_id'),
                'processing_time': form_data_for_post.get(f'job_{i}_processing_time'),
                'due_date': form_data_for_post.get(f'job_{i}_due_date'),
            })

    objective_choice=request.GET.get(
            'objective_choice') if request.method == 'GET' else form_data_for_post.get('objective_choice')
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'single_machine_demo',
        'jobs_list': jobs_list,  # 가공된 리스트 전달
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_jobs_options': range(2, 11),
        'submitted_num_jobs': submitted_num_jobs,
        'plot_data': None,
        'objective_options': [
            {'value': 'total_flow_time', 'name': '총 흐름 시간 최소화 (SPT)'},
            {'value': 'makespan', 'name': '총 완료 시간 최소화 (Makespan)'},
            {'value': 'total_tardiness', 'name': '총 지연 시간 최소화'}
        ],
        # objective_choice도 form_data 대신 직접 전달
        'submitted_objective': objective_choice
    }

    if request.method == 'POST':
        logger.info(f"Single Machine Demo POST received. Objective: {objective_choice}")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_single_machine_json_data(jobs_list, form_data_for_post, submitted_num_jobs)
            
            # 2. 파일 저장
            saved_filename, save_error = save_production_json_data(input_data)
            if save_error:
                context['error_message'] = save_error
            elif saved_filename:
                context['info_message'] = f"입력 데이터가 '{saved_filename}'으로 서버에 저장되었습니다."

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time_ms = run_single_machine_optimizer(input_data)
            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적 스케줄 계산 완료! 목표값: {results_data['objective_value']:.2f}"

                # 간트 차트용 데이터 준비
                plot_data = {'jobs': []}
                for job in results_data['schedule']:
                    plot_data['jobs'].append({
                        'label': job['id'],
                        'data': [job['start'], job['end']]  # [시작시간, 종료시간]
                    })
                context['plot_data'] = json.dumps(plot_data)

        except (ValueError, TypeError) as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'production_scheduling_app/single_machine_demo.html', context)


def single_machine_advanced_view(request):
    """Single Machine Scheduling Advanced Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'single_machine',
        'active_submenu': 'single_machine_advanced'
    }
    logger.debug("Rendering Single Machine Scheduling advanced page.")
    return render(request, 'production_scheduling_app/single_machine_advanced.html', context)


def flow_shop_introduction_view(request):
    """Flow Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'flow_shop',
        'active_submenu': 'flow_shop_introduction'
    }
    logger.debug("Rendering Flow Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/flow_shop_introduction.html', context)


def flow_shop_demo_view(request):
    """Flow Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'flow_shop',
        'active_submenu': 'flow_shop_demo'
    }
    logger.debug("Rendering Flow Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/flow_shop_demo.html', context)


def job_shop_introduction_view(request):
    """Job Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'job_shop',
        'active_submenu': 'job_shop_introduction'
    }
    logger.debug("Rendering Job Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/job_shop_introduction.html', context)


def job_shop_demo_view(request):
    """Flow Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'job_shop',
        'active_submenu': 'job_shop_demo'
    }
    logger.debug("Rendering Flow Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/job_shop_demo.html', context)


def rcpsp_introduction_view(request):
    """RCPSP Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'rcpsp',
        'active_submenu': 'rcpsp_introduction'
    }
    logger.debug("Rendering RCPSP introduction page.")
    return render(request, 'production_scheduling_app/rcpsp_introduction.html', context)


def rcpsp_demo_view(request):
    """Flow Shop Scheduling Introduction Page."""
    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu_category': 'rcpsp',
        'active_submenu': 'rcpsp_demo'
    }
    logger.debug("Rendering Flow Shop Scheduling introduction page.")
    return render(request, 'production_scheduling_app/rcpsp_demo.html', context)
