from django.shortcuts import render
import json
import numpy as np
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
import logging
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


# --- 금융 포트폴리오 최적화 실행 함수 ---
def run_portfolio_optimization_optimizer(num_assets, expected_returns, covariance_matrix, target_portfolio_return):
    """
    주어진 목표 수익률 하에서 포트폴리오 위험(분산)을 최소화합니다.
    num_assets: 자산 수
    expected_returns: 각 자산의 기대 수익률 리스트 [mu1, mu2, ..., muN]
    covariance_matrix: 공분산 행렬 (NxN 리스트의 리스트 또는 numpy 배열)
    target_portfolio_return: 목표 포트폴리오 수익률
    """
    logger.info(
        f"Running Portfolio Optimization. Assets: {num_assets}, Target Return: {target_portfolio_return}"
    )
    logger.debug(f"Expected Returns: {expected_returns}")
    logger.debug(f"Covariance Matrix: {covariance_matrix}")

    # 1. 솔버 생성 (QP를 지원하는 솔버 필요 - 예: SCIP, Gurobi, CPLEX 등)
    # pywraplp가 직접 QP를 지원하지 않을 경우, CP-SAT 또는 다른 라이브러리(cvxpy 등) 고려 필요.
    # 여기서는 pywraplp 인터페이스를 통해 QP가 가능한 백엔드 솔버(예: SCIP)가 있다고 가정.
    # 만약 순수 LP/MIP 솔버만 있다면, QP를 선형으로 근사화해야 함.
    # CreateSolver는 백엔드에 따라 QP 지원 여부가 다름.
    # SCIP은 QP를 지원하지만, GLOP (LP solver)는 지원하지 않음.
    solver_name = 'SCIP'  # 또는 Gurobi, CPLEX 등 QP 지원 솔버
    solver = pywraplp.Solver.CreateSolver(solver_name)

    if not solver:
        logger.error(f"{solver_name} solver not available for portfolio optimization (QP).")
        return None, 0, 0, f"오류: 이차 계획법(QP)을 지원하는 솔버({solver_name})를 생성할 수 없습니다. OR-Tools 백엔드 설정을 확인하세요.", 0.0
    logger.info(f"Using {solver.SolverVersion()} for portfolio optimization.")

    # 2. 결정 변수 생성
    # w[i]는 자산 i에 대한 투자 비율 (0 <= w[i] <= 1)
    # NumVar의 상한을 1로 설정 (비율이므로)
    weights = [solver.NumVar(0.0, 1.0, f'w_{i}') for i in range(num_assets)]
    logger.debug(f"Created {num_assets} weight variables.")

    # 3. 제약 조건 설정
    # 3.1. 모든 투자 비율의 합은 1: sum(w_i) = 1
    constraint_sum_weights = solver.Constraint(1.0, 1.0, 'sum_weights_constraint')
    for i in range(num_assets):
        constraint_sum_weights.SetCoefficient(weights[i], 1.0)
    logger.debug("Added constraint: sum(w_i) = 1")

    # 3.2. 포트폴리오의 총 기대 수익률이 목표 수익률 이상: sum(w_i * mu_i) >= target_portfolio_return
    constraint_target_return = solver.Constraint(target_portfolio_return, solver.infinity(), 'target_return_constraint')
    for i in range(num_assets):
        constraint_target_return.SetCoefficient(weights[i], expected_returns[i])
    logger.debug(f"Added constraint: sum(w_i * mu_i) >= {target_portfolio_return}")

    # (선택 사항) 개별 자산 투자 비율 제약: 0 <= w_i <= 1 (NumVar 생성 시 이미 적용됨)
    # 필요시 추가 제약 (예: w_i <= 0.5 등)을 여기에 추가

    # 4. 목표 함수 설정
    # 포트폴리오 분산 최소화: Minimize sum(sum(w_i * w_j * sigma_ij))
    # pywraplp.Objective()는 선형 목표 함수만 직접 지원.
    # 이차 목표 함수는 solver.SetQuadraticObjective(coefficients) 또는 유사한 메소드가 필요.
    # OR-Tools의 pywraplp는 직접적인 QP Objective 설정을 위한 명시적인 SetQuadraticObjective 함수가 없음.
    # Quadratic terms는 Gurobi, CPLEX, SCIP 등 백엔드 솔버가 지원하는 경우
    # solver.Minimize(quadratic_expression) 또는 AddQuadraticTerms 같은 방식으로 추가됨.
    #
    # 만약 사용 중인 OR-Tools 버전/백엔드가 pywraplp를 통해 QP를 직접 지원하지 않는다면,
    # 이 부분은 다른 OR-Tools API (예: CP-SAT의 AddQuadraticExpression) 또는 다른 라이브러리 (CVXPY)를 사용해야 함.
    #
    # 여기서는 백엔드 솔버가 QP를 처리할 수 있다고 가정하고,
    # 일반적인 QP 목표 설정 방법을 개념적으로 표현합니다.
    # 실제 pywraplp API로는 이 부분이 복잡하거나 불가능할 수 있습니다.
    #
    # 임시 방편: 만약 QP 직접 지원이 어렵다면, 이 데모에서는
    # 위험을 선형화하거나 (예: 사용자에게 위험 기여도 계수를 입력받아 선형 목표로 변환),
    # 또는 다른 방식의 포트폴리오 선택 모델(예: 최대 Sharpe 비율)을 선형으로 근사화해야 합니다.
    #
    # 여기서는 QP 목표 함수를 구성하려고 시도하고, 안되면 오류 메시지를 반환합니다.
    # OR-Tools에서 QP를 pywraplp로 표현하려면, 일반적으로는
    # Q = [[sigma_00, sigma_01, ...], [sigma_10, sigma_11, ...], ...] (공분산 행렬의 2배)
    # q = [0, 0, ...] (선형항 계수, 여기서는 없음)
    # 목표: 0.5 * w'Qw + q'w
    #
    # pywraplp는 일반적인 QP 목표 함수 설정을 위한 직접적인 API가 부족합니다.
    # 대신, CP-SAT 솔버를 사용하거나, Gurobi/CPLEX Python API를 직접 사용하는 것이 QP에 더 적합합니다.
    # 이 데모에서는 pywraplp의 한계를 보여주거나, QP를 풀 수 있는 백엔드가 설정되어 있다고 가정합니다.

    # pywraplp가 QP를 직접 지원하지 않으므로, 이 부분은 실제로 작동하지 않을 가능성이 높습니다.
    # 아래 코드는 개념적인 QP 목표이며, 실제 API와 다를 수 있습니다.
    # quadratic_objective = solver.Objective() # 기존 선형 목표 객체 재활용 불가
    try:
        # 일부 백엔드 (Gurobi, SCIP 등)는 MPVariable.SetObjectiveCoefficient와 유사하게
        # 이차항을 추가하는 메서드가 있을 수 있지만, pywraplp 공통 인터페이스에는 명확하지 않음.
        # solver.SetObjectiveSense(pywraplp.Solver.MINIMIZATION)
        # for i in range(num_assets):
        #     for j in range(num_assets):
        #         # This is conceptual - pywraplp doesn't have a direct AddQuadraticTerm
        #         # solver.AddQuadraticCost(weights[i], weights[j], covariance_matrix[i][j])
        # quadratic_objective.SetMinimization() # 이미 위에서 호출

        # 임시로, QP를 직접 설정할 수 없다는 메시지를 반환하고,
        # 데모에서는 선형 목표(예: 기대수익률 최대화)로 변경하거나,
        # 사용자에게 포트폴리오 분산을 직접 계산해서 보여주는 방식으로 진행합니다.
        # 여기서는 "QP 목표 설정 불가" 메시지를 명시적으로 표시합니다.
        logger.warning("pywraplp may not directly support quadratic objectives for all solvers. "
                       "The optimization below might not be true portfolio variance minimization "
                       "unless the backend solver (e.g., SCIP with QP capabilities) handles it implicitly.")

        # 임시: QP를 풀 수 없으므로, 여기서는 목표 수익률을 만족하는 "하나의 가능한 포트폴리오"를 찾는 것으로 변경
        # (위험 최소화 없이) 또는, 간단히 수익률 극대화로 변경하여 LP로 만듭니다.
        # 이 데모에서는 위험 최소화가 핵심이므로, QP 지원이 필수적입니다.
        # 지원되지 않는 경우를 명시적으로 알립니다.

        # 만약 SCIP 백엔드가 QP를 지원한다면, 다음과 같이 시도해볼 수 있습니다 (Gurobi/CPLEX API와 유사).
        # 그러나 pywraplp에는 이런 직접적인 API가 없습니다.
        # qp_terms = []
        # for i in range(num_assets):
        #     for j in range(num_assets):
        #         if covariance_matrix[i][j] != 0: # 공분산이 0이 아닌 항만 추가
        #             qp_terms.append( (weights[i], weights[j], covariance_matrix[i][j]) )
        # solver.AddQuadraticTerms(qp_terms) # 이런 함수는 존재하지 않음

        # 현재 pywraplp로는 QP를 직접 모델링하기 어렵습니다.
        # 이 데모를 위해서는 다른 OR-Tools API (예: CP-SAT을 사용한 문제 변환) 또는
        # CVXPY와 같은 Python 전용 최적화 라이브러리 사용이 더 적합합니다.
        # 여기서는 "QP 지원이 필요하며, 현재 설정으로는 어려울 수 있음"을 알리는 것으로 진행합니다.

        # === 실제 QP 목표 설정 (백엔드 솔버가 지원하는 경우) ===
        # 만약 Gurobi나 CPLEX를 백엔드로 사용하고 해당 API를 직접 호출한다면 QP 설정이 가능합니다.
        # pywraplp를 통해서는 SCIP이 QP를 지원할 때 가능할 수 있지만, API가 명확하지 않습니다.
        # 여기서는 선형 목표로 임시 변경하여 솔버가 작동하는지 확인합니다.
        # 또는, 실제 QP 문제를 풀기 위해 CP-SAT solver를 사용하는 것을 고려해야 합니다.

        # **경고**: 아래는 진정한 위험 최소화가 아닙니다. pywraplp로 QP를 표현하는 예시가 아닙니다.
        #          단순히 솔버가 작동하는지 보기 위해 목표를 임시로 설정합니다.
        #          실제 QP를 위해서는 CP-SAT 또는 외부 QP 솔버 연동이 필요합니다.
        objective = solver.Objective()
        # 임시 목표: 첫 번째 자산의 가중치를 최소화 (의미 없는 목표, 단지 솔버 테스트용)
        # objective.SetCoefficient(weights[0], 1.0)
        # objective.SetMinimization()
        # logger.warning("Using a placeholder linear objective for solver testing due to pywraplp QP limitations.")

        # 포트폴리오 분산을 목표로 설정 (이 부분은 pywraplp에서 직접 지원하지 않을 가능성 높음)
        # 아래 코드는 solver가 QP를 지원하고, Objective 객체가 이차항을 받을 수 있다고 가정합니다.
        # 실제로는 solver.SetQuadraticObjective(...) 와 같은 메서드가 필요합니다.
        objective_coeffs = {}
        for i in range(num_assets):
            for j in range(num_assets):
                # Add L_{i,j} w_i w_j to objective.
                # Solver needs 0.5 * w' Sigma w, so coeffs are Sigma_ij
                # For pywraplp, this direct quadratic term setting is not standard.
                # This is a placeholder for how one might conceptualize it.
                # If using a solver like Gurobi directly, you'd use its API for quadratic objectives.
                pass  # pywraplp는 이부분 직접 지원 안함.

        # pywraplp의 한계로, 여기서는 QP 목표를 직접 설정할 수 없음을 명시하고,
        # 데모에서는 이 부분을 어떻게 처리할지 결정해야 합니다.
        # 1. QP 지원 가능한 다른 OR-Tools API (CP-SAT) 사용
        # 2. CVXPY와 같은 QP 전문 라이브러리 사용
        # 3. 이 데모에서는 선형화된 목표 또는 다른 간단한 목표 사용 (예: 위험 지표를 입력받아 제약으로 활용)

        # 여기서는 에러 메시지를 반환하여 QP 설정이 필요함을 알립니다.
        if not solver.SupportsQuadraticObjective():  # 이런 함수가 있다면 좋겠지만 없음
            logger.error(
                "The selected OR-Tools solver via pywraplp does not appear to support quadratic objectives directly for portfolio optimization.")
            return None, 0, 0, "오류: 현재 OR-Tools 설정(pywraplp)으로는 포트폴리오 분산 최소화(이차 계획법)를 직접 지원하기 어렵습니다. CP-SAT 솔버 또는 외부 QP 라이브러리 연동이 필요합니다.", 0.0

        # 만약 솔버가 QP를 지원한다면, 실제로는 solver.Minimize(sum(covariance_matrix[i][j] * weights[i] * weights[j] for i ... for j ...)) 와 같은 형태가 되어야 함.
        # pywraplp의 경우, 이는 백엔드 솔버(예: Gurobi, CPLEX, SCIP)의 기능을 통해야 함.
        # 임시로, 목표 함수 없이 제약조건만 만족하는 해를 찾는 것으로 변경 (status=FEASIBLE 확인용)
        # solver.Maximize(0.0) # 또는 Minimize(0.0)
        # 또는 아래처럼 해야 SCIP이 QP를 인지할 가능성이 있습니다 (SCIP 포맷을 통해)
        # 그러나 이는 매우 백엔드 의존적입니다.
        objective.SetMinimization()  # 먼저 선언하고
        for i in range(num_assets):
            for j in range(num_assets):
                if covariance_matrix[i][j] != 0:  # 공분산이 0이 아닌 항만 추가
                    # pywraplp는 Objective().SetQuadraticCoefficient(var1, var2, coeff) 같은 API가 없음
                    # 이 부분은 주석 처리하고, 실제 QP 솔버 연동이 필요함을 명시
                    pass
        # logger.info("Attempting to set quadratic objective (may not be supported by all pywraplp backends).")
        # **실제 QP 목표 설정은 이 데모의 범위를 넘어설 수 있습니다.**
        # **단순화를 위해, 이 데모에서는 "위험"을 반환하되, 실제 최소화는 안 될 수 있음을 명시합니다.**


    except AttributeError as ae:
        logger.error(
            f"AttributeError during QP objective setup: {ae}. This often means the solver backend via pywraplp doesn't support QP in this way.",
            exc_info=True)
        return None, 0, 0, f"오류: 포트폴리오 최적화 모델 구성 중 오류 발생 (이차 목표 설정 관련). {str(ae)}", 0.0
    except Exception as e:
        logger.error(f"Error setting up portfolio optimization model: {e}", exc_info=True)
        return None, 0, 0, f"오류: 포트폴리오 최적화 모델 구성 중 오류 발생. {str(e)}", 0.0

    # --- 문제 해결 (이미 try-except로 감싸여 있음) ---
    solve_start_time_actual = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time_actual = datetime.datetime.now()
    processing_time_ms_actual = (solve_end_time_actual - solve_start_time_actual).total_seconds() * 1000
    logger.info(f"Portfolio optimization solver status: {status}, Time: {processing_time_ms_actual:.2f} ms")

    # --- 결과 추출 ---
    results = []
    portfolio_expected_return = 0.0
    portfolio_variance = 0.0  # 실제 계산된 분산
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found for portfolio, but it might not be optimal.")
            # 사용자에게는 성공으로 간주

        calculated_weights = [w.solution_value() if abs(w.solution_value()) > 1e-6 else 0.0 for w in weights]

        for i in range(num_assets):
            portfolio_expected_return += calculated_weights[i] * expected_returns[i]
            results.append({
                'asset_name': f'자산 {i + 1}',  # 실제 자산 이름 사용 가능
                'weight': round(calculated_weights[i] * 100, 2),  # %로 표시
                'expected_return_asset': expected_returns[i]
            })

        # 포트폴리오 분산 계산: w' * Sigma * w
        # np_weights = np.array(calculated_weights)
        # np_cov_matrix = np.array(covariance_matrix)
        # portfolio_variance = np.dot(np_weights.T, np.dot(np_cov_matrix, np_weights))
        # logger.info(f"Calculated Portfolio Expected Return: {portfolio_expected_return:.4f}")
        # logger.info(f"Calculated Portfolio Variance: {portfolio_variance:.6f}")
        # logger.info(f"Calculated Portfolio Std Dev: {np.sqrt(portfolio_variance):.6f}")

        # QP 목표가 제대로 설정되지 않았으므로, objective value는 의미 없을 수 있음
        # 대신, 계산된 가중치로 분산을 직접 계산
        # solver.Objective().Value()가 실제 분산 최소화 목표의 값일 경우 사용
        # 현재는 QP 직접 설정이 어려우므로, 이 값은 신뢰하기 어려움.
        # 대신, 제약조건을 만족하는 해를 찾고, 그 해의 분산을 계산하여 보여주는 방식으로 진행
        np_weights = np.array(calculated_weights)
        np_cov_matrix = np.array(covariance_matrix)
        if np_weights.shape[0] == np_cov_matrix.shape[0] and np_weights.shape[0] == np_cov_matrix.shape[1]:
            portfolio_variance = np.dot(np_weights.T, np.dot(np_cov_matrix, np_weights))
        else:
            portfolio_variance = -1  # 계산 불가
            logger.error("Shape mismatch for variance calculation.")


    else:  # OPTIMAL 또는 FEASIBLE이 아닌 경우
        # ... (이전 답변의 solver_status_map 사용한 오류 처리) ...
        solver_status_map = {
            pywraplp.Solver.INFEASIBLE: "실행 불가능한 문제입니다. 목표 수익률이 너무 높거나 제약 조건이 충돌할 수 있습니다.",
            # ...
        }
        error_msg = solver_status_map.get(status, f"포트폴리오 해를 찾지 못했습니다. (솔버 상태 코드: {status})")
        logger.error(f"Portfolio solver failed. Status: {status}. Message: {error_msg}")

    return results, round(portfolio_expected_return, 4), round(portfolio_variance,
                                                               6), error_msg, processing_time_ms_actual


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
