from django.conf import settings
from django.shortcuts import render
import json
import itertools
import random

from common_utils.run_production_opt import *
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
    form_data ={}
    periods_data = []  # 템플릿에 전달할 데이터 리스트
    if request.method == 'GET':
        submitted_num_periods = int(request.GET.get('num_periods_to_show', preset_lot_sizing_num_periods))
        submitted_num_periods = max(3, min(12, submitted_num_periods))

        # GET 요청 시 랜덤 기본값으로 form_data 초기화
        for t in range(submitted_num_periods):
            preset = preset_lot_sizing_data[t]
            periods_data.append({
                'demand': request.GET.get(f'demand_{t}', preset['demand']),
                'setup_cost': request.GET.get(f'setup_cost_{t}', preset['setup_cost']),
                'prod_cost': request.GET.get(f'prod_cost_{t}', preset['prod_cost']),
                'holding_cost': request.GET.get(f'holding_cost_{t}', preset['holding_cost']),
                'capacity': request.GET.get(f'capacity_{t}', preset['capacity']),
            })
        # periods_data.append({
        #     'demand': request.GET.get(f'demand_{t}', str(random.randint(50, 150))),
        #     'setup_cost': request.GET.get(f'setup_cost_{t}', str(random.randint(200, 500))),
        #     'prod_cost': request.GET.get(f'prod_cost_{t}', str(random.randint(5, 15))),
        #     'holding_cost': request.GET.get(f'holding_cost_{t}', str(random.randint(1, 5))),
        #     'capacity': request.GET.get(f'capacity_{t}', str(random.randint(150, 300)))
        # })
        logger.info(periods_data)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_periods = int(form_data.get('num_periods', preset_lot_sizing_num_periods))

        # POST 후 입력값 유지를 위해 periods_data 재생성
        for t in range(submitted_num_periods):
            preset = preset_lot_sizing_data[t]
            periods_data.append({
                'demand': form_data.get(f'demand_{t}', preset['demand']),
                'setup_cost': form_data.get(f'setup_cost_{t}', preset['setup_cost']),
                'prod_cost': form_data.get(f'prod_cost_{t}', preset['prod_cost']),
                'holding_cost': form_data.get(f'holding_cost_{t}', preset['holding_cost']),
                'capacity': form_data.get(f'capacity_{t}', preset['capacity']),
            })

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'lot_sizing_demo',
        'periods_data': periods_data, # 구조화된 리스트 전달
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
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_production_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_lot_sizing_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

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

    if request.method == 'GET':
        submitted_num_jobs = int(request.GET.get('num_jobs_to_show', preset_single_machine_num_jobs))
        submitted_objective = request.GET.get('objective_choice', preset_single_machine_objective_choice)
        submitted_num_jobs = max(2, min(10, submitted_num_jobs))

        # GET 요청 시, URL 파라미터 또는 기본 데이터 풀 값으로 jobs_list 구성
        for i in range(submitted_num_jobs):
            preset = preset_single_machine_data[i]
            jobs_list.append({
                'id': request.GET.get(f'job_{i}_id', preset['id']),
                'processing_time': request.GET.get(f'job_{i}_processing_time', preset['processing_time']),
                'due_date': request.GET.get(f'job_{i}_due_date', preset['due_date']),
                'release_time': request.GET.get(f'job_{i}_release_time', preset['release_time']),
            })

    elif request.method == 'POST':
        form_data  = request.POST
        submitted_num_jobs = int(form_data .get('num_jobs', preset_single_machine_num_jobs))
        submitted_objective = form_data.get('objective_choice', preset_single_machine_objective_choice)

        for i in range(submitted_num_jobs):
            jobs_list.append({
                'id': form_data.get(f'job_{i}_id'),
                'processing_time': form_data.get(f'job_{i}_processing_time'),
                'due_date': form_data.get(f'job_{i}_due_date'),
                'release_time': form_data.get(f'job_{i}_release_time'),
            })

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'single_machine_demo',
        'jobs_list': jobs_list,  # 구조화된 리스트 전달
        'results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_jobs_options': range(2, 11),
        'submitted_num_jobs': submitted_num_jobs,
        'plot_data': {},
        'objective_options': preset_single_machine_objective,
        'submitted_objective': submitted_objective
    }

    if request.method == 'POST':
        logger.info(f"Single Machine Demo POST received. Objective: {submitted_objective}")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_single_machine_json_data(jobs_list, submitted_objective , submitted_num_jobs)
            
            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_production_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_single_machine_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data:
                context['results'] = results_data
                context['success_message'] = (f"최적 스케줄 계산 완료! 최종 완료일: {results_data['makespan']},"
                                              f" 총 지연 시간 합계: {results_data['total_tardiness']}, "
                                              f" 흐름 시간 합계: {results_data['total_flow_time']}")
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
    jobs_list = []
    if request.method == 'GET':
        submitted_num_jobs = int(request.GET.get('num_jobs_to_show', preset_flow_shop_num_jobs))
        submitted_num_machines = int(request.GET.get('num_machines_to_show', preset_flow_shop_num_machines))
        submitted_num_jobs = max(2, min(8, submitted_num_jobs))
        submitted_num_machines = max(3, min(5, submitted_num_machines))

        # GET 요청 시, URL 파라미터 또는 기본 데이터 풀 값으로 jobs_list 구성
        for i in range(submitted_num_jobs):
            preset = preset_flow_shop_data[i]
            job_info = {
                'id': request.GET.get(f'job_{i}_id', preset['id']),
                'processing_times': []
            }
            for j in range(submitted_num_machines):
                time_val = request.GET.get(f'p_{i}_{j}', preset['processing_times'][j])
                job_info['processing_times'].append(time_val)
            jobs_list.append(job_info)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_jobs = int(form_data.get('num_jobs', preset_flow_shop_num_jobs))
        submitted_num_machines = int(form_data.get('num_machines', preset_flow_shop_num_machines))

        # POST 후 입력값 유지를 위해 jobs_list 재생성
        for i in range(submitted_num_jobs):
            job_info = {
                'id': form_data.get(f'job_{i}_id'),
                'processing_times': [form_data.get(f'p_{i}_{j}') for j in range(submitted_num_machines)]
            }
            jobs_list.append(job_info)

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'flow_shop_demo',
        'jobs_list': jobs_list,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_jobs_options': range(2, 11),
        'num_machines_options': range(3, 6),
        'submitted_num_jobs': submitted_num_jobs,
        'submitted_num_machines': submitted_num_machines,
        'plot_data': None
    }

    if request.method == 'POST':
        action = request.POST.get('action')
        try:
            if action == "optimize":    # 최적화 실행 버튼
                logger.info(f"Flow Shop Demo {action} POST request processing.")
                # 1. 데이터 파일 새성 및 검증
                input_data = create_flow_shop_json_data(form_data)

                # 2. 파일 저장
                if settings.SAVE_DATA_FILE:
                    success_save_message, save_error = save_production_json_data(input_data)
                    if save_error:
                        context['error_message'] = save_error
                    elif success_save_message:
                        context['success_save_message'] = success_save_message

                # 3. 최적화 실행
                results_data, error_msg_opt, processing_time = run_flow_shop_optimizer(input_data)
                context['processing_time_seconds'] = processing_time

                if error_msg_opt:
                    context['error_message'] = error_msg_opt
                elif results_data:
                    context['results'] = results_data
                    context['success_message'] = f"최적 스케줄 계산 완료! Makespan: {results_data['makespan']}"
                    # 간트 차트용 데이터 준비
                    context['plot_data'] = json.dumps(results_data['schedule'])
                    # 중요한 부분: 다음 수동 조회를 위해 원본 데이터와 결과를 숨겨진 필드로 전달할 수 있도록 저장
                    context['original_input_json'] = json.dumps(input_data)
                    context['optimal_results_json'] = json.dumps(results_data)
            elif action == 'manual_check':  # 수동 순서 조회 버튼
                logger.info(f"Flow Shop Demo {action} POST request processing.")

                # 숨겨진 필드에서 원본 데이터와 최적화 결과 로드
                original_input_str = form_data.get('original_input_json')
                optimal_results_str = form_data.get('optimal_results_json')
                if not original_input_str or not optimal_results_str:
                    raise ValueError("비교를 위한 원본 데이터 또는 최적화 결과가 없습니다.")

                original_input_data = json.loads(original_input_str)
                optimal_results_data = json.loads(optimal_results_str)
                context['results'] = optimal_results_data  # 최적 결과 다시 표시
                context['plot_data'] = json.dumps(optimal_results_data['schedule'])
                context['original_input_json'] = original_input_str
                context['optimal_results_json'] = optimal_results_str

                # 사용자가 입력한 수동 시퀀스 파싱
                manual_sequence_str = form_data.get('manual_sequence', '')
                manual_sequence = [s.strip() for s in manual_sequence_str.split(',') if s.strip()]

                # 수동 시퀀스로 스케줄 계산
                manual_results_data = calculate_flow_shop_schedule(
                    original_input_data['processing_times'],
                    original_input_data['job_ids'],
                    manual_sequence
                )
                context['manual_results'] = manual_results_data
                context['manual_plot_data'] = json.dumps(manual_results_data['schedule'])
                context['info_message'] \
                    = f"수동 입력 순서 '{', '.join(manual_sequence)}'의 Makespan은 {manual_results_data['makespan']} 입니다."
        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

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
    jobs_list = []

    if request.method == 'GET':
        submitted_num_jobs = int(request.GET.get('num_jobs_to_show', preset_job_shop_num_jobs))
        submitted_num_machines = int(request.GET.get('num_machines_to_show', preset_job_shop_num_machines))

        # GET 요청 시, URL 파라미터 또는 기본 데이터 풀 값으로 jobs_list 구성
        for i in range(submitted_num_jobs):
            preset = preset_job_shop_data[i]
            job_info = {
                'id': request.GET.get(f'job_{i}_id', preset['id']),
                'processing_times': [],
                'selected_routing': request.GET.get(f'job_{i}_routing', preset['selected_routing'])
            }
            for j in range(submitted_num_machines):
                time_val = request.GET.get(f'p_{i}_{j}', preset['processing_times'][j])
                job_info['processing_times'].append(time_val)
            jobs_list.append(job_info)

    elif request.method == 'POST':
        form_data_for_repopulate = request.POST.copy()
        submitted_num_jobs = int(form_data_for_repopulate.get('num_jobs', preset_job_shop_num_jobs))
        submitted_num_machines = int(form_data_for_repopulate.get('num_machines', preset_job_shop_num_machines))

        # POST 후 입력값 유지를 위해 jobs_list 재생성
        for i in range(submitted_num_jobs):
            job_info = {
                'id': form_data_for_repopulate.get(f'job_{i}_id'),
                'processing_times': [],
                'selected_routing': form_data_for_repopulate.get(f'job_{i}_routing')
            }
            for j in range(submitted_num_machines):
                job_info['processing_times'].append(form_data_for_repopulate.get(f'p_{i}_{j}'))
            jobs_list.append(job_info)

    # 가능한 공정 순서(라우팅) 목록 생성
    machine_indices = list(range(submitted_num_machines))
    possible_routings = list(itertools.permutations(machine_indices))

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'job_shop_demo',
        'jobs_list': jobs_list,  # 평평한 form_data 대신 구조화된 리스트 전달
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_jobs_options': range(2, 6),
        'num_machines_options': range(3, 5),
        'submitted_num_jobs': submitted_num_jobs,
        'submitted_num_machines': submitted_num_machines,
        'possible_routings': possible_routings,
        'plot_data': None
    }

    if request.method == 'POST':
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_job_shop_json_data(form_data_for_repopulate)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_production_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_job_shop_optimizer(input_data)
            context['processing_time_seconds'] = processing_time
            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적 스케줄 계산 완료! Makespan: {results_data['makespan']}"
                context['plot_data'] = json.dumps(results_data['schedule'])
        except (ValueError, TypeError) as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

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
    """
    Resource-Constrained Project Scheduling Problem (RCPSP) 데모 뷰.
    """
    activities_list = []
    resources_list=[]

    # GET 요청 시: URL 파라미터 또는 기본값으로 form_data 초기화
    if request.method == 'GET':
        submitted_num_activities = int(request.GET.get('num_activities_to_show', preset_rcpsp_num_activities))
        submitted_num_resources = int(request.GET.get('num_resources_to_show', preset_rcpsp_num_resources))
        submitted_num_activities = max(3, min(8, submitted_num_activities))
        submitted_num_resources = max(1, min(3, submitted_num_resources))

        # 가용 자원 기본값
        for k in range(submitted_num_resources):
            preset = preset_rcpsp_resource_data[k]
            resources_list.append({
                'name': request.GET.get(f'resource_{k}_name', preset['name']),
                'availability': request.GET.get(f'resource_{k}_availability', preset['availability'])
            })

        for i in range(submitted_num_activities):
            preset = preset_rcpsp_activities_data[i]
            activity_info = {
                'id': request.GET.get(f'activity_{i}_id', preset['id']),
                'duration': request.GET.get(f'activity_{i}_duration', preset['duration']),
                'predecessors': request.GET.get(f'activity_{i}_predecessors', preset['predecessors']),
                'res_reqs': []
            }
            for k in range(submitted_num_resources):
                req = request.GET.get(f'activity_{i}_res_{k}_req',
                                      str(preset['res_reqs'][k] if k < len(preset['res_reqs']) else 0))
                activity_info['res_reqs'].append(req)
            activities_list.append(activity_info)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_activities = int(form_data.get('num_activities', preset_rcpsp_num_activities))
        submitted_num_resources = int(form_data.get('num_resources', preset_rcpsp_num_resources))

        # POST 후 입력값 유지를 위해 리스트 재생성
        for k in range(submitted_num_resources):
            resources_list.append({
                'name': form_data.get(f'resource_{k}_name'),
                'availability': form_data.get(f'resource_{k}_availability')
            })

        for i in range(submitted_num_activities):
            activity_info = {
                'id': form_data.get(f'activity_{i}_id'),
                'duration': form_data.get(f'activity_{i}_duration'),
                'predecessors': form_data.get(f'activity_{i}_predecessors'),
                'res_reqs': []
            }
            for k in range(submitted_num_resources):
                activity_info['res_reqs'].append(form_data.get(f'activity_{i}_res_{k}_req'))
            activities_list.append(activity_info)

    context = {
        'active_model': 'Production & Scheduling',
        'active_submenu': 'rcpsp_demo',
        'resources_list': resources_list,
        'activities_list': activities_list,
        'results': None, 'error_message': None, 'success_message': None, 'info_message': None,
        'processing_time_seconds': "N/A",
        'num_activities_options': range(3, 9),
        'num_resources_options': range(1, 4),
        'submitted_num_activities': submitted_num_activities,
        'submitted_num_resources': submitted_num_resources,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info("RCPSP Demo POST request processing.")
        try:
            input_data = create_rcpsp_json_data(form_data)

            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_production_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_rcpsp_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적 프로젝트 스케줄 수립 완료! 최소 프로젝트 기간(Makespan): {results_data['makespan']}일"
                # 차트용 데이터 준비
                chart_data = {
                    'schedule': results_data['schedule'],
                    'resource_usage': results_data['resource_usage'],
                    'resource_availabilities': input_data['resource_availabilities']
                }
                context['plot_data'] = json.dumps(chart_data)
            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'production_scheduling_app/rcpsp_demo.html', context)