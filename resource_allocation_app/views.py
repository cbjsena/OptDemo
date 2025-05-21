from django.shortcuts import render
import json
import random
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
import logging
import os
from django.conf import settings  # settings.LARGE_SCALE_DATA_DIR 사용을 위함
import datetime  # 파일명 생성 등에 사용 가능

logger = logging.getLogger(__name__)  # settings.py에 정의된 'resource_allocation_app' 로거 사용


# --- 공통 유효성 검사 함수 ---
def validate_panel_data_structure(panel_list_items, panel_type_name):
    """
    패널 데이터 리스트의 구조와 내용의 유효성을 검사합니다. (Matching 앱에서 사용되던 함수)
    이 앱(resource_allocation_app)에서는 현재 직접 사용되지 않지만, 필요시 유사한 함수를 만들 수 있습니다.
    여기서는 예산 분배 문제에 맞는 유효성 검사가 각 뷰 함수 내에 구현됩니다.
    """
    if panel_list_items is None:
        return f"오류: '{panel_type_name}_panels' 데이터가 없습니다 (None)."
    if not isinstance(panel_list_items, list):
        return f"오류: '{panel_type_name}_panels' 데이터가 리스트 형식이 아닙니다."
    if not panel_list_items:
        return f"오류: '{panel_type_name}_panels' 리스트가 비어있습니다."

    for p_idx, p_item in enumerate(panel_list_items):
        if not isinstance(p_item, dict):
            return f"오류: {panel_type_name} 패널 데이터 (인덱스 {p_idx})가 딕셔너리 형식이 아닙니다."

        required_keys = ('id', 'rows', 'cols', 'defect_map')  # 예시 키 (Matching 문제용)
        missing_keys = [k for k in required_keys if k not in p_item]
        if missing_keys:
            return f"오류: {panel_type_name} 패널 (ID: {p_item.get('id', 'N/A')}, 인덱스 {p_idx})에 필수 키가 누락되었습니다: {', '.join(missing_keys)}."
        # ... (이하 Matching 문제에 특화된 유효성 검사는 이 앱에서는 불필요)
    return None


# --- 최적 예산 분배 알고리즘 ---
def run_budget_allocation_optimizer(total_budget, items_data):
    logger.info(f"Running budget allocation for Total Budget: {total_budget}, Items: {len(items_data)}")

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        logger.error("GLOP Solver not available for budget allocation.")
        return None, 0, "오류: 선형 계획법 솔버(GLOP)를 생성할 수 없습니다.", 0.0

    num_items = len(items_data)
    infinity = solver.infinity()

    x = [solver.NumVar(0, infinity, f'x_{i}') for i in range(num_items)]
    logger.debug(f"Created {num_items} decision variables for budget allocation.")

    # 총 예산 제약
    constraint_total_budget = solver.Constraint(0, float(total_budget), 'total_budget_constraint')
    for i in range(num_items):
        constraint_total_budget.SetCoefficient(x[i], 1)
    logger.debug(f"Added total budget constraint: sum(x_i) <= {total_budget}")

    # 개별 항목 투자 한도 제약
    for i in range(num_items):
        item = items_data[i]
        # 입력 단계에서 float으로 변환되었음을 가정, 여기서 한 번 더 확인 및 변환
        min_alloc = float(item.get('min_alloc', 0))
        max_alloc = float(item.get('max_alloc', infinity))

        # 변수 생성 시 이미 하한이 0으로 설정되었으므로, min_alloc을 적용하려면 변수 하한을 직접 수정하거나 제약조건 추가
        x[i].SetLb(min_alloc)  # 변수의 Lower Bound 설정
        x[i].SetUb(max_alloc)  # 변수의 Upper Bound 설정
        logger.debug(f"Item {item.get('name', i)} constraints: {min_alloc} <= x_{i} <= {max_alloc}")

    objective = solver.Objective()
    for i in range(num_items):
        item = items_data[i]
        return_coeff = float(item.get('return_coefficient', 0))
        objective.SetCoefficient(x[i], return_coeff)
    objective.SetMaximization()
    logger.debug("Budget allocation objective function set for maximization.")

    logger.info("Solving the budget allocation model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver status: {status}, Time: {processing_time_ms:.2f} ms")

    results = []
    total_maximized_return = 0.0
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found for budget allocation, but it might not be optimal.")
            # 사용자에게는 성공으로 알리고, 내부적으로만 경고
            # error_msg = "최적해일 수도 있지만, 더 좋은 해가 있을 가능성이 있습니다." # 이 메시지는 혼란을 줄 수 있어 제거

        raw_objective_value = solver.Objective().Value()
        total_maximized_return = raw_objective_value if raw_objective_value is not None else 0.0
        logger.info(f"Budget allocation objective value (Total Maximized Return): {total_maximized_return}")

        for i in range(num_items):
            allocated_val = x[i].solution_value()
            # 매우 작은 값은 0으로 처리 (부동소수점 정밀도 문제)
            if abs(allocated_val) < 1e-6: allocated_val = 0.0

            results.append({
                'name': items_data[i].get('name', f'항목 {i + 1}'),
                'allocated_budget': round(allocated_val, 2),
                'expected_return': round(allocated_val * float(items_data[i].get('return_coefficient', 0)), 2),
                'min_alloc': items_data[i].get('min_alloc'),
                'max_alloc': items_data[i].get('max_alloc'),
                'return_coefficient': items_data[i].get('return_coefficient')
            })
    else:
        solver_status_map = {
            pywraplp.Solver.INFEASIBLE: "실행 불가능한 문제입니다. 제약 조건을 확인하세요 (예: 총 예산이 모든 항목의 최소 투자액 합보다 작거나, 최소/최대 투자 한도 충돌).",
            # ... (이전 답변의 나머지 상태 메시지들) ...
            pywraplp.Solver.UNBOUNDED: "목표 함수가 무한합니다. 제약 조건이 누락되었을 수 있습니다.",
            pywraplp.Solver.ABNORMAL: "솔버가 비정상적으로 종료되었습니다.",
            pywraplp.Solver.NOT_SOLVED: "솔버가 문제를 풀지 못했습니다.",
            pywraplp.Solver.MODEL_INVALID: "모델이 유효하지 않습니다. 변수나 제약 조건 설정을 확인하세요."
        }
        error_msg = solver_status_map.get(status, f"최적해를 찾지 못했습니다. (솔버 상태 코드: {status})")
        logger.error(f"Budget allocation solver failed. Status: {status}. Message: {error_msg}")

    return results, total_maximized_return, error_msg, processing_time_ms


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
            items_data = []
            for i in range(num_items):
                name = form_data_from_post.get(f'item_name_{i}', f'항목 {i + 1}')
                return_coeff_str = form_data_from_post.get(f'item_return_coeff_{i}')
                min_alloc_str = form_data_from_post.get(f'item_min_alloc_{i}', '0')
                max_alloc_str = form_data_from_post.get(f'item_max_alloc_{i}', str(total_budget))

                if not return_coeff_str:
                    raise ValueError(f"'{name}'의 기대 수익률 계수가 입력되지 않았습니다.")

                try:
                    return_coeff = float(return_coeff_str)
                    min_alloc = float(min_alloc_str)
                    max_alloc = float(max_alloc_str)
                except ValueError:
                    raise ValueError(f"'{name}'의 숫자 입력값(수익률, 최소/최대 투자액)이 올바르지 않습니다.")

                if min_alloc < 0 or max_alloc < 0:
                    raise ValueError(f"'{name}'의 최소/최대 투자액은 음수가 될 수 없습니다.")
                if min_alloc > max_alloc:
                    raise ValueError(f"'{name}'의 최소 투자액({min_alloc})이 최대 투자액({max_alloc})보다 클 수 없습니다.")

                items_data.append({
                    'name': name,
                    'return_coefficient': return_coeff,
                    'min_alloc': min_alloc,
                    'max_alloc': max_alloc
                })

            logger.debug(f"Parsed items_data for optimizer: {items_data}")

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

                context['success_message'] = "최적 예산 분배 계산 완료!"
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


def financial_portfolio_demo_view():
    return None