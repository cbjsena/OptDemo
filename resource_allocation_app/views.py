from django.shortcuts import render
from django.conf import settings
import json

from common_utils.nurseRosteringSolver import NurseRosteringSolver
from common_utils.data_utils_allocation import *
from common_utils.run_alloctaion_opt import *

logger = logging.getLogger(__name__)  # settings.py에 정의된 'resource_allocation_app' 로거 사용


def resource_allocation_introduction_view(request):
    """General introduction page for the Resource Allocation category."""
    context = {
        'active_model': 'Resource Allocation',
        # 이 페이지는 특정 소메뉴에 속하지 않으므로 active_submenu는 비워둠
        'active_submenu': 'main_introduction'
    }
    logger.debug("Rendering general Resource & Allocation introduction page.")
    return render(request, 'resource_allocation_app/resource_allocation_introduction.html', context)


def resource_allocation_view():
    return None


def budget_allocation_introduction_view(request):
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'budget_allocation_introduction'
    }
    logger.debug("Rendering budget allocation introduction page.")
    return render(request, 'resource_allocation_app/budget_allocation_introduction.html', context)

def budget_allocation_demo_view(request):
    # 요청 방식에 따라 데이터 소스 결정
    source = request.POST if request.method == 'POST' else request.GET

    # --- 어떤 요청이든 현재 설정값을 읽어옴 ---
    submitted_num_items = int(source.get('num_items_to_show', source.get('num_items', preset_budget_num_item)))
    total_budget = int(source.get('total_budget', preset_total_budget))

    items_data = []
    for i in range(submitted_num_items):
        preset = preset_budget_items[i]
        items_data.append({
            'name': source.get(f'item_{i}_name', preset['name']),
            'return_coefficient': source.get(f'item_{i}_return_coefficient', preset['return_coefficient']),
            'min_alloc': source.get(f'item_{i}_min_alloc', preset['min_alloc']),
            'max_alloc': source.get(f'item_{i}_max_alloc', preset['max_alloc']),
        })

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'Budget Allocation Demo',
        'num_items_options': range(2, 11),
        'total_budget': total_budget,
        'items_data': items_data,
        'results': None,
        'processing_time_seconds': "N/A",
        'error_message': None,
        'success_message': None,
        'submitted_num_items': submitted_num_items
    }

    if request.method == 'POST':
        logger.info("Budget allocation demo POST request received.")
        try:
            # 1. 데이터 생성 및 검증
            input_data = create_budget_allocation_json_data(source)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_allocation_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message
            logger.info( context.get('success_save_message', ''))
            # 3. 최적화 실행
            results, error_msg, processing_time = run_budget_allocation_optimizer(input_data)
            context['processing_time_seconds'] = processing_time
            if error_msg:
                context['error_message'] = error_msg
            else:
                context['results'] = results
                context['success_message'] = f"최적 예산 분배 수립 완료! 최대 기대 수익: {results['total_maximized_return']}"
                logger.info(
                    f"Budget allocation successful. Max return: {results['total_maximized_return']}, "
                    f"Total allocated: {results['total_allocated_budget']}, "
                    f"Utilization: {results['budget_utilization_percent']}%")

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in budget_allocation_demo_view: {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in budget_allocation_demo_view: {e}", exc_info=True)

    return render(request, 'resource_allocation_app/budget_allocation_demo.html', context)


def financial_portfolio_introduction_view(request):
    """
    Financial Portfolio Optimization introduction page.
    """
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'financial_portfolio_introduction'
    }
    logger.debug("Rendering financial portfolio optimization introduction page.")
    return render(request, 'resource_allocation_app/financial_portfolio_introduction.html', context)


def financial_portfolio_demo_view(request):
    # Default values for GET request
    default_num_assets = 3
    default_target_return = '0.10'  # 10%
    default_asset_returns = [0.12, 0.10, 0.15, 0.08, 0.11]  # Up to 5 assets
    default_asset_stddevs = [0.20, 0.15, 0.25, 0.12, 0.18]  # Up to 5 assets

    current_form_data = {}
    submitted_num_assets_val = default_num_assets

    if request.method == 'GET':
        num_assets_from_get = request.GET.get('num_assets_to_show')
        if num_assets_from_get:
            try:
                num_to_show = int(num_assets_from_get)
                if 2 <= num_to_show <= 5:  # Assuming max 5 for this example
                    submitted_num_assets_val = num_to_show
            except ValueError:
                pass  # Use default

        current_form_data['target_portfolio_return'] = request.GET.get('target_portfolio_return', default_target_return)
        for i in range(submitted_num_assets_val):
            current_form_data[f'asset_name_{i}'] = request.GET.get(f'asset_name_{i}', f'자산 {i + 1}')
            current_form_data[f'asset_return_{i}'] = request.GET.get(f'asset_return_{i}', str(
                default_asset_returns[i] if i < len(default_asset_returns) else 0.1))
            current_form_data[f'asset_stddev_{i}'] = request.GET.get(f'asset_stddev_{i}', str(
                default_asset_stddevs[i] if i < len(default_asset_stddevs) else 0.15))

    elif request.method == 'POST':
        current_form_data = request.POST.copy()
        submitted_num_assets_val = int(current_form_data.get('num_assets', default_num_assets))

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'Financial Portfolio Demo',
        'num_assets_options': range(2, 6),  # 예: 2~5개 자산
        'form_data': current_form_data,
        'results': None,
        'portfolio_metrics': None,
        'error_message': None,
        'success_message': None,
        'submitted_num_assets': submitted_num_assets_val  # 기본 자산 수
    }

    if request.method == 'POST':
        logger.info("Financial Portfolio demo POST request received.")

        try:
            target_return_str = current_form_data.get('target_portfolio_return')
            if not target_return_str:
                raise ValueError("목표 포트폴리오 수익률이 입력되지 않았습니다.")
            target_portfolio_return = float(target_return_str)

            expected_returns = []
            # 공분산 행렬 생성을 위해 표준편차와 상관계수 입력받도록 단순화 (또는 직접 공분산 행렬 입력)
            # 이 데모에서는 표준편차만 입력받고, 상관관계는 없다고 가정 (공분산 행렬의 대각 성분만 존재)
            # 또는 모든 상관계수를 0.5 등으로 가정하여 공분산 행렬 생성
            asset_stddevs = []

            for i in range(submitted_num_assets_val):
                ret_str = current_form_data.get(f'asset_return_{i}')
                std_str = current_form_data.get(f'asset_stddev_{i}')  # 표준편차 입력

                if not ret_str or not std_str:
                    raise ValueError(f"'자산 {i+1}'의 기대 수익률 또는 표준편차(위험)가 입력되지 않았습니다.")

                expected_returns.append(float(ret_str))
                asset_stddevs.append(float(std_str))

            # 실제로는 사용자가 상관계수나 전체 공분산 행렬을 입력하도록 해야 함.
            # 공분산 행렬 생성 (예시: 모든 자산 간 상관계수를 0.3으로 가정)
            correlation_coeff = 0.3
            cov_matrix_np = np.zeros((submitted_num_assets_val, submitted_num_assets_val))
            for i in range(submitted_num_assets_val):
                for j in range(submitted_num_assets_val):
                    if i == j:
                        cov_matrix_np[i, j] = asset_stddevs[i] ** 2  # Variance = StdDev^2
                    else:
                        cov_matrix_np[i, j] = correlation_coeff * asset_stddevs[i] * asset_stddevs[j]
            covariance_matrix = cov_matrix_np.tolist()

            logger.debug(
                f'Data for portfolio optimizer: ER={expected_returns}, COV={covariance_matrix}, '
                f'TargetRet={target_portfolio_return}')

            results, calc_portfolio_return, calc_portfolio_variance, error_msg, processing_time = \
                run_portfolio_optimization_optimizer(submitted_num_assets_val, expected_returns, covariance_matrix,
                                                     target_portfolio_return)
            context['processing_time_seconds'] = processing_time
            if error_msg:
                context['error_message'] = error_msg
            elif results is not None:
                context['results'] = results
                portfolio_std_dev = np.sqrt(
                    calc_portfolio_variance) if calc_portfolio_variance >= 0 else -1.0  # Ensure float for N/A case or calculation

                context['portfolio_metrics'] = {
                    'expected_return': round(calc_portfolio_return * 100, 2),
                    'variance': round(calc_portfolio_variance, 6),
                    'std_dev': round(portfolio_std_dev * 100, 2) if portfolio_std_dev >= 0 else "N/A",
                    'target_return_input': round(target_portfolio_return * 100, 2)
                }
                # Assign asset names to results here, if not done in run_portfolio_optimization_optimizer
                for i, res_item in enumerate(results):
                    res_item['asset_name'] = current_form_data.get(f'asset_name_{i}', f'자산 {i + 1}')

                context['success_message'] = "포트폴리오 최적화 계산 완료!"
                logger.info(
                    f"Portfolio optimization successful. ER: {calc_portfolio_return}, Var: {calc_portfolio_variance}")
            else:
                context['error_message'] = "최적화 결과를 가져오는 데 실패했습니다 (결과 없음)."
        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in financial_portfolio_demo_view: {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in financial_portfolio_demo_view: {e}", exc_info=True)

    return render(request, 'resource_allocation_app/financial_portfolio_demo.html', context)


def data_center_capacity_introduction_view(request): # Renamed to be more general for the sub-menu
    """
    Capacity Investment & Allocation (Data Center Capacity Planning) introduction page.
    """
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'data_center_capacity_introduction' # Key for the specific intro page
    }
    logger.debug("Rendering data center capacity planning introduction page.")
    return render(request, 'resource_allocation_app/data_center_capacity_introduction.html', context)


def data_center_capacity_demo_view(request):
    source = request.POST if request.method == 'POST' else request.GET

    submitted_num_server_types = int(
        source.get('num_server_types_to_show', source.get('num_server_types', preset_datacenter_num_server_types)))
    submitted_num_services = int(source.get('num_services_to_show', source.get('num_services', preset_datacenter_num_services)))

    global_constraints = {
        'total_budget': source.get('total_budget', 100000),
        'total_power_kva': source.get('total_power_kva', 50),
        'total_space_sqm': source.get('total_space_sqm', 10),
    }

    servers_data = []
    for i in range(submitted_num_server_types):
        preset = preset_datacenter_servers[i % len(preset_datacenter_servers)]
        servers_data.append({
            'id': source.get(f'server_{i}_id', preset.get('id')),
            'cost': source.get(f'server_{i}_cost', preset.get('cost')),
            'cpu_cores': source.get(f'server_{i}_cpu_cores', preset.get('cpu_cores')),
            'ram_gb': source.get(f'server_{i}_ram_gb', preset.get('ram_gb')),
            'storage_tb': source.get(f'server_{i}_storage_tb', preset.get('storage_tb')),
            'power_kva': source.get(f'server_{i}_power_kva', preset.get('power_kva')),
            'space_sqm': source.get(f'server_{i}_space_sqm', preset.get('space_sqm')),
        })

    services_data = []
    for i in range(submitted_num_services):
        preset = preset_datacenter_services[i % len(preset_datacenter_services)]
        services_data.append({
            'id': source.get(f'service_{i}_id', preset.get('id')),
            'revenue_per_unit': source.get(f'service_{i}_revenue_per_unit', preset.get('revenue_per_unit')),
            'req_cpu_cores': source.get(f'service_{i}_req_cpu_cores', preset.get('req_cpu_cores')),
            'req_ram_gb': source.get(f'service_{i}_req_ram_gb', preset.get('req_ram_gb')),
            'req_storage_tb': source.get(f'service_{i}_req_storage_tb', preset.get('req_storage_tb')),
            'max_units': source.get(f'service_{i}_max_units', preset.get('max_units')),
        })

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'Data Center Capacity Demo',
        'global_constraints': global_constraints,
        'servers_data': servers_data,
        'services_data': services_data,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_server_types_options': range(1, 4),
        'num_services_options': range(1, 4),
        'submitted_num_server_types': submitted_num_server_types,
        'submitted_num_services': submitted_num_services,
        'chart_data_py': {},
        'chart_data_json': None
    }

    if request.method == 'POST':
        logger.info("Data Center Capacity Demo POST processing.")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_datacenter_allocation_json_data(source)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_allocation_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_datacenter_capacity_optimizer(input_data)
            context['processing_time_seconds'] = processing_time
            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f'데이터 센터 용량 계획 최적화 완료!'.strip()
                logger.info(
                    f"Data center capacity optimization successful. Total Profit: {results_data.get('total_profit')}")

                # --- 차트 데이터 준비 ---
                chart_data_py_dict  = set_datacenter_chart_data(results_data, input_data.get('global_constraints'))
                context['chart_data_py'] = chart_data_py_dict
                context['chart_data_json'] = json.dumps(chart_data_py_dict )  # JSON 문자열로 템플릿에 전달
            else:
                context['error_message'] = (context.get('error_message', '') + " 최적화 결과를 가져오지 못했습니다 (결과 없음).").strip()
        except ValueError as ve: # 파싱 및 유효성 검사 오류
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in data_center_capacity_demo_view (POST): {ve}", exc_info=True)
        except Exception as e:  # 그 외 모든 예외
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in data_center_capacity_demo_view (POST): {e}", exc_info=True)

    return render(request, 'resource_allocation_app/data_center_capacity_demo.html', context)

# --- 3. Nurse Rostering Problem ---
def nurse_rostering_introduction_view(request):
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'nurse_rostering_introduction'
    }
    logger.debug("Rendering Nurse Rostering introduction page.")
    return render(request, 'resource_allocation_app/nurse_rostering_introduction.html', context)


def nurse_rostering_demo_view(request):
    source = request.POST if request.method == 'POST' else request.GET
    submitted_num_nurses = int(source.get('num_nurses', preset_nurse_rostering_num_nurses))
    submitted_num_days = int(source.get('num_days', preset_nurse_rostering_days))
    submitted_min_shifts = int(source.get('min_shifts', preset_nurse_rostering_min_shifts))
    submitted_max_shifts = int(source.get('max_shifts', preset_nurse_rostering_max_shifts))
    shift_requests = preset_nurse_rostering_requests

    if request.method == 'POST':
        shift_requests_parsed = {}
        for s_idx, s_name in enumerate(preset_nurse_rostering_shifts):
            required = int(source.get(f'shift_{s_idx}_req'))
            for d in range(submitted_num_days):
                shift_requests_parsed[(d, s_idx)] = required

    schedule_weekdays = get_schedule_weekdays(preset_nurse_rostering_days)
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'Nurse Rostering Demo',
        'shifts': preset_nurse_rostering_shifts,
        'shift_requests': shift_requests,
        'schedule_weekdays': schedule_weekdays,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        # 설정 옵션
        'num_nurses_options': preset_nurse_num_nurses_options,
        'num_days_options': preset_nurse_num_days_options,
        'min_shifts_options': preset_nurse_min_shifts_options,
        'max_shifts_options': preset_nurse_max_shifts_options,
        # 현재 선택된 값
        'submitted_num_nurses': submitted_num_nurses,
        'submitted_num_days': submitted_num_days,
        'submitted_min_shifts': submitted_min_shifts,
        'submitted_max_shifts': submitted_max_shifts,
    }

    if request.method == 'POST':
        try:
            input_data = create_nurse_rostering_json_data(source)

            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_allocation_json_data(input_data)
                if save_error:
                    context['error_message'] = (
                                context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 최적화 실행
            results_data, error_msg, processing_time = run_nurse_rostering_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg:
                context['error_message'] = error_msg
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적의 근무표를 생성했습니다! (총 페널티: {results_data['total_penalty']})"

        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
    logger.debug("Rendering Nurse Rostering demo page.")
    return render(request, 'resource_allocation_app/nurse_rostering_demo.html', context)


def nurse_rostering_advanced_demo_view(request):
    logger.info(f"Start nurse_rostering_advanced_demo_view.")

    # 요청 방식에 따라 데이터 소스 결정
    source = request.POST if request.method == 'POST' else request.GET
    submitted_num_nurses = int(source.get('num_nurses', preset_nurse_rostering_num_nurses))
    submitted_num_days = int(source.get('num_days', preset_nurse_rostering_days))
    submitted_min_shifts = int(source.get('min_shifts', preset_nurse_rostering_min_shifts))
    submitted_max_shifts = int(source.get('max_shifts', preset_nurse_rostering_max_shifts))
    nurses_data = []
    for i in range(submitted_num_nurses):
        preset = preset_nurse_rostering_nurses_data[i]
        nurses_data.append({
            'id': i,
            'name': source.get(f'nurse_{i}_name', preset['name']),
            'skill': source.get(f'nurse_{i}_skill', preset['skill']),
        })

    # --- 시프트별 필요인원, 휴가, 공정성 옵션 값 유지 ---
    skill_requirements = {}
    for s_name in preset_nurse_rostering_shifts:
        skill_requirements[s_name] = {}
        for skill in preset_nurse_rostering_skill_options:
            default_req = preset_nurse_rostering_shift_requirements.get(s_name).get(skill, '0')
            skill_requirements[s_name][skill] = int(source.get(f'req_{s_name}_{skill}', default_req))

    submitted_vacations = {i: source.get(f'nurse_{i}_vacation', '') for i in range(submitted_num_nurses)}
    submitted_fairness = source.getlist('fairness_options')
    if request.method != 'POST' and not submitted_fairness:  # 최초 로드 시 기본값 체크
        submitted_fairness = preset_nurse_rostering_enabled_fairness

    schedule_weekdays = get_schedule_weekdays(preset_nurse_rostering_days)

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'Advanced Nurse Rostering',
        'nurses_data': nurses_data,
        'shifts': preset_nurse_rostering_shifts,
        'skill_options': preset_nurse_rostering_skill_options,
        'schedule_weekdays': schedule_weekdays,
        'skill_requirements': skill_requirements,
        'submitted_vacations': submitted_vacations,
        'submitted_fairness': submitted_fairness,
        'results': None, 'error_message': None, 'success_message': None, 'success_save_message': None,
        'processing_time_seconds': "N/A",
        # 설정 옵션
        'num_nurses_options': preset_nurse_num_nurses_options,
        'num_days_options': preset_nurse_num_days_options,
        'min_shifts_options': preset_nurse_min_shifts_options,
        'max_shifts_options': preset_nurse_max_shifts_options,
        # 현재 선택된 값
        'submitted_num_nurses': submitted_num_nurses,
        'submitted_num_days': submitted_num_days,
        'submitted_min_shifts': submitted_min_shifts,
        'submitted_max_shifts': submitted_max_shifts,
    }

    if request.method == 'POST':
        try:
            # 1. 데이터 생성 및 검증
            input_data = create_nurse_rostering_advanced_json_data(source)

            # 2. 파일 저장 (선택 사항)
            if settings.SAVE_DATA_FILE:
                # save_allocation_json_data 함수가 있다고 가정
                success_save_message, save_error = save_allocation_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            # results_data, error_msg_opt, processing_time = run_nurse_roster_advanced_optimizer(input_data)
            results_data, error_msg_opt, processing_time = NurseRosteringSolver(input_data).solve()
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적의 근무표를 생성했습니다! (총 페널티: {results_data['total_penalty']})"

        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'resource_allocation_app/nurse_rostering_advanced_demo.html', context)


