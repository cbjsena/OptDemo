from django.shortcuts import render
import numpy as np
import logging
import json

from .utils import data_utils
from common_utils.run_alloctaion_opt import *
from common_utils.default_data import (
    budjet_items_preset,
    datacenter_servers_preset,
    datacenter_services_preset
)
logger = logging.getLogger(__name__)  # settings.py에 정의된 'resource_allocation_app' 로거 사용


def budget_allocation_introduction_view(request):
    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'budget_allocation_introduction'
    }
    logger.debug("Rendering budget allocation introduction page.")
    return render(request, 'resource_allocation_app/budget_allocation_introduction.html', context)


def budget_allocation_demo_view(request):
    # Initialize form_data with a default total_budget for GET requests
    # This ensures the template's default_if_none has something to fall back on
    # or that the key exists.
    initial_form_data = {'total_budget': '1000'}  # Default value for initial page load
    form_data = {}
    for i in range(len(budjet_items_preset)):
        preset = budjet_items_preset[i % len(budjet_items_preset)]
        for key, default_val in preset.items():
            initial_form_data[key] = default_val

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'budget_allocation_demo',
        'num_items_options': range(1, 11),
        'form_data': initial_form_data.copy(),  # Use a copy for GET
        'results': None,
        'total_maximized_return': None,
        'total_allocated_budget': 0,
        'budget_utilization_percent': "N/A",
        'processing_time_seconds': "N/A",
        'error_message': None,
        'success_message': None,
        'submitted_num_items': 3
    }

    if request.method == 'GET':
        if 'num_items_to_show' in request.GET:
            try:
                num_to_show = int(request.GET.get('num_items_to_show', 3))
                if not (1 <= num_to_show <= 10):
                    num_to_show = 3
                context['submitted_num_items'] = num_to_show
                logger.debug(f"Number of items to show set to: {num_to_show} via GET param.")
            except ValueError:
                logger.warning("Invalid num_items_to_show in GET param, using default.")
                context['submitted_num_items'] = 3
        # For GET, context['form_data'] will retain initial_form_data
        # If you want to persist form values across GET requests (e.g., after num_items change),
        # you'd need to pass them in the GET params and repopulate form_data here.
        # For now, initial_form_data provides a default.

    if request.method == 'POST':
        logger.info("Budget allocation demo POST request received.")
        form_data_from_post = request.POST.copy()
        context['form_data'] = form_data_from_post  # Overwrite with POST data
        context['submitted_num_items'] = int(form_data_from_post.get('num_items', 3))

        try:
            total_budget_str = form_data_from_post.get('total_budget')
            if not total_budget_str:
                raise ValueError("총 예산이 입력되지 않았습니다.")
            total_budget = float(total_budget_str)
            if total_budget < 0:
                raise ValueError("총 예산은 음수가 될 수 없습니다.")
            # Keep total_budget as float for calculations
            # context['form_data']['total_budget_float'] = total_budget # Not strictly needed if form_data['total_budget'] is used carefully

            num_items = context['submitted_num_items']
            items_data, json_data = data_utils.parse_allocation_budjet_data(form_data_from_post, num_items, total_budget)
            logger.debug(f"Parsed items_data for optimizer: {items_data}")

            saved_filename, save_error = data_utils.save_allocation_budjet_json_data(json_data)
            if save_error:
                context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
            elif saved_filename:
                context['success_save_message'] = (
                            context.get('info_message', '') + f" 입력 데이터가 '{saved_filename}'으로 서버에 저장.").strip()

            results, total_maximized_return, error_msg, processing_time_ms = run_budget_allocation_optimizer(
                total_budget, items_data)

            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg:
                context['error_message'] = error_msg
            else:
                context['results'] = results
                context['total_maximized_return'] = round(total_maximized_return, 2)

                calculated_total_allocated = 0
                if results:
                    for item_result in results:
                        calculated_total_allocated += item_result.get('allocated_budget', 0)
                context['total_allocated_budget'] = round(calculated_total_allocated, 2)

                if total_budget > 0:
                    utilization_percent = (calculated_total_allocated / total_budget) * 100
                    context['budget_utilization_percent'] = round(utilization_percent, 1)
                else:
                    context[
                        'budget_utilization_percent'] = 0.0 if calculated_total_allocated == 0 else "N/A (Total Budget is 0)"
                current_success = context.get('info_message', "")  # 파일 저장 성공 메시지가 있다면 이어붙임
                context['success_message'] = f'최적 예산 분배 계산 완료!'.strip()
                logger.info(
                    f"Budget allocation successful. Max return: {total_maximized_return}, Total allocated: {calculated_total_allocated}, Utilization: {context['budget_utilization_percent']}%")

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
        'active_submenu': 'financial_portfolio_demo',
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
                f"Data for portfolio optimizer: ER={expected_returns}, COV={covariance_matrix}, TargetRet={target_portfolio_return}")

            results, calc_portfolio_return, calc_portfolio_variance, error_msg, processing_time_ms = \
                run_portfolio_optimization_optimizer(submitted_num_assets_val, expected_returns, covariance_matrix,
                                                     target_portfolio_return)

            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

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
    default_num_server_types = 2
    default_num_services = 2
    submitted_num_server_types= default_num_server_types
    submitted_num_services = default_num_services
    # GET 요청 시 또는 POST 후 폼 값 유지를 위한 form_data 초기화
    form_data = {}

    # GET 요청 시 기본 폼 데이터 채우기
    if request.method == 'GET':
        # URL 파라미터 또는 기본값으로 항목 수 결정
        submitted_num_server_types = int(request.GET.get('num_server_types_to_show', default_num_server_types))
        submitted_num_services = int(request.GET.get('num_services_to_show', default_num_services))

        # 범위 제한
        submitted_num_server_types = max(1, min(3, submitted_num_server_types))
        submitted_num_services = max(1, min(3, submitted_num_services))

        # 기본 글로벌 제약 조건 값 설정
        form_data['total_budget'] = request.GET.get('total_budget', 100000)
        form_data['total_power_kva'] = request.GET.get('total_power_kva', 50)
        form_data['total_space_sqm'] = request.GET.get('total_space_sqm', 10)
        # 기본 서버 유형 데이터 (ID 포함)
        for i in range(submitted_num_server_types):
            preset = datacenter_servers_preset[i % len(datacenter_servers_preset)]
            for key, default_val in preset.items():
                form_data[f'server_{i}_{key}'] = request.GET.get(f'server_{i}_{key}', default_val)

        # 기본 서비스 수요 데이터 (ID 포함)
        for i in range(submitted_num_services):
            preset = datacenter_services_preset[i % len(datacenter_services_preset)]
            for key, default_val in preset.items():
                form_data[f'service_{i}_{key}'] = request.GET.get(f'service_{i}_{key}', default_val)
    elif request.method == 'POST':
        form_data = request.POST.copy()  # POST 요청 시에는 제출된 데이터 사용
        submitted_num_server_types = int(form_data.get('num_server_types', default_num_server_types))
        submitted_num_services = int(form_data.get('num_services', default_num_services))

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu': 'data_center_capacity_demo',
        'form_data': form_data, # GET 또는 POST로부터 채워진 form_data
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
        'num_server_types_options': range(1, 4),
        'num_services_options': range(1, 4),
        'submitted_num_server_types': submitted_num_server_types,
        'submitted_num_services': submitted_num_services,
        # 템플릿의 if 조건문에서는 Python 딕셔너리 객체를 기준으로 조건을 확인
        # JavaScript에서 사용할 때는 JSON 문자열을 파싱해서 객체
        'chart_data_py': {},
        'chart_data_json':None
    }
    if request.method == 'POST':
        logger.info("Data Center Capacity Demo POST processing.")
        try:
            # --- 1. 입력 데이터 파싱---
            parsed_global_constraints, parsed_server_types_data, parsed_service_demands_data = data_utils.parse_allocation_data_center_data(
                form_data, submitted_num_server_types, submitted_num_services
            )

            # --- 2. 유효성 검사 ---
            validation_error = data_utils.validate_data_center_data(
                parsed_global_constraints,  # 원본 수정을 피하기 위해 복사본 전달
                parsed_server_types_data,
                parsed_service_demands_data
            )
            if validation_error:
                raise ValueError(validation_error)  # 유효성 검사 실패 시 ValueError 발생

            # --- 3. 입력 데이터 JSON 파일로 저장 ---
            input_data = data_utils.create_allocation_data_center_json_data(
                parsed_global_constraints,
                parsed_server_types_data,
                parsed_service_demands_data
            )
            saved_filename, save_error = data_utils.save_allocation_data_center_json_data(input_data)
            if save_error:
                context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
            elif saved_filename:
                context['success_save_message'] = (
                            context.get('info_message', '') + f" 입력 데이터가 '{saved_filename}'으로 서버에 저장.").strip()

            # --- 4. 최적화 실행 ---
            results_data, error_msg_opt, processing_time_ms = run_data_center_capacity_optimizer(
                parsed_global_constraints, parsed_server_types_data, parsed_service_demands_data
            )
            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f'데이터 센터 용량 계획 최적화 완료!'.strip()
                logger.info(
                    f"Data center capacity optimization successful. Total Profit: {results_data.get('total_profit')}")

                # --- 차트 데이터 준비 ---
                chart_data_py_dict  = data_utils.set_chart_data(results_data, parsed_global_constraints)
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
