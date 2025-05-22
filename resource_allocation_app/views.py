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


def run_data_center_capacity_optimizer(global_constraints, server_types_data, service_demands_data):
    """
    데이터 센터 용량 계획 문제를 해결합니다.
    global_constraints: {'total_budget': 100000, 'total_power_kva': 500, 'total_space_sqm': 200}
    server_types_data: [{'id': 'S1', 'cost': 5000, 'cpu_cores': 64, 'ram_gb': 256, 'storage_tb': 10, 'power_kva': 0.5, 'space_sqm': 0.2}, ...]
    service_demands_data: [{'id': 'WEB', 'revenue_per_unit': 100, 'req_cpu_cores': 8, 'req_ram_gb': 32, 'req_storage_tb': 1, 'max_units': 50}, ...]
    """
    logger.info("Running Data Center Capacity Optimizer...")
    logger.debug(f"Global Constraints: {global_constraints}")
    logger.debug(f"Server Types: {server_types_data}")
    logger.debug(f"Service Demands: {service_demands_data}")

    solver = pywraplp.Solver.CreateSolver('CBC')  # 또는 'SCIP' 등 MIP 지원 솔버
    if not solver:
        logger.error("CBC/SCIP MIP Solver not available for data center capacity planning.")
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    infinity = solver.infinity()
    num_server_types = len(server_types_data)
    num_services = len(service_demands_data)

    # --- 결정 변수 ---
    # Ns[i]: 구매/설치할 서버 유형 i의 수 (정수 변수)
    Ns = [solver.IntVar(0, infinity, f'Ns_{i}') for i in range(num_server_types)]

    # As[j][i]: 서비스 j에 할당된 서버 유형 i에서 제공되는 서비스 단위의 수 (연속 또는 정수 변수)
    # 이 데모에서는 서비스 단위를 '서버가 제공하는 용량 단위'로 단순화하고,
    # 각 서비스가 특정 서버 유형에서 몇 개의 '인스턴스' 또는 '용량 블록'을 사용하는지로 모델링 가능.
    # 더 간단하게는, 서비스 j를 위해 사용되는 서버 i의 "비율" 또는 "개수"로 볼 수 있음.
    # 여기서는 서비스 j를 위해 사용되는 서버 유형 i의 개수 (Ns_i 중 일부)로 가정.
    # 이는 너무 복잡해지므로, 총 가용 용량을 계산하고 서비스에 할당하는 방식으로 변경.

    # Xsj: 서비스 s를 서버 유형 j에 몇 개나 배치할 것인가 (또는 서비스 s에 서버 j의 자원을 얼마나 할당할 것인가)
    # 이 데모에서는 "서비스 유닛"이라는 추상적인 단위를 사용.
    # U sj: 서비스 s의 유닛 j (0 to max_units_s-1) - 사용 안함.

    # As_sj: 서비스 s에 할당된 서버 유형 j의 용량 단위 수 (예: CPU 코어 수)
    # 더 명확한 모델:
    # N_server[i]: 구매할 서버 i의 개수 (정수)
    # Alloc_service_server[s][i]: 서비스 s를 위해 서버 i에 할당된 "자원 단위" 또는 "서비스 인스턴스 수"
    # 여기서는, 각 서비스가 특정 양의 CPU, RAM, Storage를 요구하고,
    # 각 서버 유형이 특정 양의 CPU, RAM, Storage를 제공한다고 가정.

    # X_si: 서비스 s를 서버 유형 i에서 몇 유닛(instance) 실행할 것인가 (정수)
    # max_units_s는 서비스 s의 최대 수요 또는 제공 가능한 최대 유닛.
    # 각 서비스 유닛은 특정 자원을 소모함.
    X_si = {}
    for s_idx in range(num_services):
        service = service_demands_data[s_idx]
        # 서비스 s의 최대 유닛 수 (수요) 만큼 변수 생성 고려
        # 또는, 총 제공 가능한 서비스 유닛을 변수로 할 수도 있음.
        # 여기서는 서비스 s를 서버 유형 i에서 몇 '유닛'만큼 제공할지를 변수로 설정.
        # 이 '유닛'은 해당 서비스의 요구 자원에 맞춰짐.
        max_units_s = service.get('max_units', infinity) if service.get('max_units') is not None else infinity
        for i_idx in range(num_server_types):
            # 서비스 s를 서버 i에서 몇 유닛 제공할지 (이산적인 서비스 유닛으로 가정)
            X_si[s_idx, i_idx] = solver.IntVar(0, max_units_s if max_units_s != infinity else solver.infinity(),
                                               f'X_s{s_idx}_i{i_idx}')

    logger.debug(f"Created {len(Ns)} Ns variables and {len(X_si)} X_si variables.")

    # --- 제약 조건 ---
    # 1. 총 예산 제약
    total_budget_constraint = solver.Constraint(0, global_constraints.get('total_budget', infinity), 'total_budget')
    for i in range(num_server_types):
        total_budget_constraint.SetCoefficient(Ns[i], server_types_data[i].get('cost', 0))
    logger.debug("Added total budget constraint.")

    # 2. 총 전력 제약
    total_power_constraint = solver.Constraint(0, global_constraints.get('total_power_kva', infinity), 'total_power')
    for i in range(num_server_types):
        total_power_constraint.SetCoefficient(Ns[i], server_types_data[i].get('power_kva', 0))
    logger.debug("Added total power constraint.")

    # 3. 총 공간 제약
    total_space_constraint = solver.Constraint(0, global_constraints.get('total_space_sqm', infinity), 'total_space')
    for i in range(num_server_types):
        total_space_constraint.SetCoefficient(Ns[i], server_types_data[i].get('space_sqm', 0))
    logger.debug("Added total space constraint.")

    # 4. 각 자원(CPU, RAM, Storage)에 대한 용량 제약
    # 각 서버 유형 i가 제공하는 총 CPU = Ns[i] * server_types_data[i]['cpu_cores']
    # 각 서비스 s의 유닛이 요구하는 CPU = service_demands_data[s]['req_cpu_cores']
    # 총 요구 CPU = sum over s,i (X_si[s,i] * service_demands_data[s]['req_cpu_cores'])
    # 이는 잘못된 접근. X_si는 서비스 s를 서버 i에서 몇 유닛 제공하는지.
    # 서버 i에 할당된 서비스들의 총 요구 자원이 서버 i의 총 제공 자원을 넘을 수 없음.

    # 수정된 제약: 각 서버 유형 i에 대해, 해당 서버 유형에 할당된 모든 서비스의 자원 요구량 합계는
    # 해당 서버 유형의 총 구매된 용량을 초과할 수 없음.
    resource_types = ['cpu_cores', 'ram_gb', 'storage_tb']
    for i_idx in range(num_server_types):  # 각 서버 유형에 대해
        server_type = server_types_data[i_idx]
        for res_idx, resource in enumerate(resource_types):  # 각 자원 유형에 대해
            # 서버 유형 i가 제공하는 총 자원량
            # Ns[i_idx] * server_type.get(resource, 0)

            # 서버 유형 i에 할당된 모든 서비스 유닛들이 소모하는 총 자원량
            # sum (X_si[s_idx, i_idx] * service_demands_data[s_idx].get(f'req_{resource}', 0) for s_idx in range(num_services))

            constraint_res = solver.Constraint(-infinity, 0, f'res_{resource}_server_type_{i_idx}')
            # 제공량 (우변으로 넘기면 <= 0)
            constraint_res.SetCoefficient(Ns[i_idx], -server_type.get(resource, 0))  # 제공량은 음수로
            # 소비량 (좌변에 그대로)
            for s_idx in range(num_services):
                if (s_idx, i_idx) in X_si:  # 해당 변수가 존재할 때만
                    service = service_demands_data[s_idx]
                    constraint_res.SetCoefficient(X_si[s_idx, i_idx], service.get(f'req_{resource}', 0))  # 소비량은 양수로
            logger.debug(f"Added resource constraint for {resource} on server type {server_type.get('id', i_idx)}.")

    # 5. 각 서비스의 최대 수요(유닛) 제약 (선택 사항, X_si 변수 상한으로 이미 반영됨)
    # sum over i (X_si[s,i]) <= service_demands_data[s]['max_units'] (또는 == nếu 정확히 수요 충족)
    for s_idx in range(num_services):
        service = service_demands_data[s_idx]
        max_units_s = service.get('max_units')
        if max_units_s is not None and max_units_s != infinity:
            # 서비스 s에 대해 모든 서버 유형에서 제공되는 총 유닛 수는 max_units_s를 넘을 수 없음
            constraint_demand_s = solver.Constraint(0, max_units_s, f'demand_service_{s_idx}')
            for i_idx in range(num_server_types):
                if (s_idx, i_idx) in X_si:
                    constraint_demand_s.SetCoefficient(X_si[s_idx, i_idx], 1)
            logger.debug(f"Added max demand constraint for service {service.get('id', s_idx)}: <= {max_units_s}")

    # --- 목표 함수 ---
    # 총 이익 = (각 서비스 유닛 수익 합계) - (총 서버 구매 비용)
    objective = solver.Objective()
    # 서버 구매 비용 (음수)
    for i in range(num_server_types):
        objective.SetCoefficient(Ns[i], -server_types_data[i].get('cost', 0))

    # 서비스 수익 (양수)
    for s_idx in range(num_services):
        service = service_demands_data[s_idx]
        for i_idx in range(num_server_types):
            if (s_idx, i_idx) in X_si:
                objective.SetCoefficient(X_si[s_idx, i_idx], service.get('revenue_per_unit', 0))

    objective.SetMaximization()
    logger.debug("Objective function set for total profit maximization.")

    # --- 문제 해결 ---
    logger.info("Solving Data Center Capacity model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver status: {status}, Time: {processing_time_ms:.2f} ms")

    # --- 결과 추출 ---
    results = {
        'purchased_servers': [],
        'service_allocations': [],
        'total_profit': 0,
        'total_server_cost': 0,
        'total_service_revenue': 0,
        'total_power_used': 0,
        'total_space_used': 0,
    }
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found for data center plan, but it might not be optimal.")
            # error_msg 설정은 선택사항

        results['total_profit'] = round(solver.Objective().Value(), 2)

        current_total_server_cost = 0
        current_total_power = 0
        current_total_space = 0
        for i in range(num_server_types):
            num_purchased = Ns[i].solution_value()
            if abs(num_purchased) < 1e-6: num_purchased = 0  # 부동소수점 정리
            num_purchased = int(round(num_purchased))  # 정수 변수이므로 반올림

            if num_purchased > 0:
                server_type = server_types_data[i]
                results['purchased_servers'].append({
                    'type_id': server_type.get('id', f'Type{i}'),
                    'count': num_purchased,
                    'unit_cost': server_type.get('cost', 0),
                    'total_cost_for_type': round(num_purchased * server_type.get('cost', 0), 2)
                })
                current_total_server_cost += num_purchased * server_type.get('cost', 0)
                current_total_power += num_purchased * server_type.get('power_kva', 0)
                current_total_space += num_purchased * server_type.get('space_sqm', 0)

        results['total_server_cost'] = round(current_total_server_cost, 2)
        results['total_power_used'] = round(current_total_power, 2)
        results['total_space_used'] = round(current_total_space, 2)

        current_total_service_revenue = 0
        service_details = []
        for s_idx in range(num_services):
            service = service_demands_data[s_idx]
            total_units_for_service_s = 0
            allocation_details_s = []
            for i_idx in range(num_server_types):
                if (s_idx, i_idx) in X_si:
                    units_on_server_i = X_si[s_idx, i_idx].solution_value()
                    if abs(units_on_server_i) < 1e-6: units_on_server_i = 0
                    units_on_server_i = int(round(units_on_server_i))

                    if units_on_server_i > 0:
                        total_units_for_service_s += units_on_server_i
                        allocation_details_s.append({
                            'server_type_id': server_types_data[i_idx].get('id', f'Type{i_idx}'),
                            'units_allocated': units_on_server_i
                        })

            if total_units_for_service_s > 0:
                service_revenue_s = total_units_for_service_s * service.get('revenue_per_unit', 0)
                current_total_service_revenue += service_revenue_s
                service_details.append({
                    'service_id': service.get('id', f'Service{s_idx}'),
                    'total_units_provided': total_units_for_service_s,
                    'revenue_from_service': round(service_revenue_s, 2),
                    'allocations': allocation_details_s
                })
        results['service_allocations'] = service_details
        results['total_service_revenue'] = round(current_total_service_revenue, 2)

        # 최종 이익 확인 (솔버 목표값과 수동 계산 일치 여부)
        manual_profit = results['total_service_revenue'] - results['total_server_cost']
        logger.info(f"Solver Objective (Profit): {results['total_profit']}, Manual Calc Profit: {manual_profit:.2f}")


    else:  # OPTIMAL 또는 FEASIBLE이 아닌 경우
        solver_status_map = {
            pywraplp.Solver.INFEASIBLE: "실행 불가능한 문제입니다. 제약 조건(예산, 전력, 공간, 자원 요구량 등)을 확인하세요.",
            pywraplp.Solver.UNBOUNDED: "목표 함수가 무한합니다. 수익이 비용보다 과도하게 높거나 제약이 누락되었을 수 있습니다.",
            pywraplp.Solver.ABNORMAL: "솔버가 비정상적으로 종료되었습니다.",
            pywraplp.Solver.MODEL_INVALID: "모델이 유효하지 않습니다.",
            pywraplp.Solver.NOT_SOLVED: "솔버가 문제를 풀지 못했습니다."
        }
        error_msg = solver_status_map.get(status, f"최적해를 찾지 못했습니다. (솔버 상태 코드: {status})")
        logger.error(f"Data center capacity solver failed. Status: {status}. Message: {error_msg}")

    return results, error_msg, processing_time_ms


def data_center_capacity_demo_view(request):
    default_num_server_types = 2
    default_num_services = 2

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
        form_data['total_budget'] = request.GET.get('total_budget', '100000')
        form_data['total_power_kva'] = request.GET.get('total_power_kva', '50')
        form_data['total_space_sqm'] = request.GET.get('total_space_sqm', '10')

        # 기본 서버 유형 데이터 (ID 포함)
        default_servers_preset = [
            {'id': 'SrvA', 'cost': '5000', 'cpu_cores': '64', 'ram_gb': '256', 'storage_tb': '10', 'power_kva': '0.5',
             'space_sqm': '0.2'},
            {'id': 'SrvB', 'cost': '3000', 'cpu_cores': '32', 'ram_gb': '128', 'storage_tb': '5', 'power_kva': '0.3',
             'space_sqm': '0.1'},
            {'id': 'SrvC', 'cost': '8000', 'cpu_cores': '128', 'ram_gb': '512', 'storage_tb': '20', 'power_kva': '0.8',
             'space_sqm': '0.3'}
        ]
        for i in range(submitted_num_server_types):
            preset = default_servers_preset[i % len(default_servers_preset)]
            for key, default_val in preset.items():
                form_data[f'server_{i}_{key}'] = request.GET.get(f'server_{i}_{key}', default_val)

            # 기본 서비스 수요 데이터 (ID 포함)
        default_services_preset = [
            {'id': 'WebPool', 'revenue_per_unit': '100', 'req_cpu_cores': '4', 'req_ram_gb': '8',
             'req_storage_tb': '0.1', 'max_units': '50'},
            {'id': 'DBFarm', 'revenue_per_unit': '200', 'req_cpu_cores': '8', 'req_ram_gb': '16',
             'req_storage_tb': '0.5', 'max_units': '20'},
            {'id': 'BatchProc', 'revenue_per_unit': '150', 'req_cpu_cores': '16', 'req_ram_gb': '32',
             'req_storage_tb': '0.2', 'max_units': '30'}
        ]
        for i in range(submitted_num_services):
            preset = default_services_preset[i % len(default_services_preset)]
            for key, default_val in preset.items():
                form_data[f'service_{i}_{key}'] = request.GET.get(f'service_{i}_{key}', default_val)

    elif request.method == 'POST':
        form_data = request.POST.copy()  # POST 요청 시에는 제출된 데이터 사용
        submitted_num_server_types = int(form_data.get('num_server_types', default_num_server_types))
        submitted_num_services = int(form_data.get('num_services', default_num_services))

    context = {
        'active_model': 'Resource Allocation',
        'active_submenu_category': 'capacity_investment_allocation',
        'active_submenu_case': 'data_center_capacity_demo',
        'form_data': form_data, # GET 또는 POST로부터 채워진 form_data
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
        'num_server_types_options': range(1, 4),
        'num_services_options': range(1, 4),
        'submitted_num_server_types': submitted_num_server_types,
        'submitted_num_services': submitted_num_services,
    }

    if request.method == 'POST':
        logger.info("Data Center Capacity Demo POST processing.")
        try:
            global_constraints = {
                'total_budget': float(form_data.get('total_budget', '0')),
                'total_power_kva': float(form_data.get('total_power_kva', '0')),
                'total_space_sqm': float(form_data.get('total_space_sqm', '0')),
            }
            if not (global_constraints['total_budget'] >= 0 and \
                    global_constraints['total_power_kva'] >= 0 and \
                    global_constraints['total_space_sqm'] >= 0):
                raise ValueError("총 예산, 전력, 공간은 음수가 될 수 없습니다.")

            server_types_data = []
            for i in range(submitted_num_server_types):
                cost_str = form_data.get(f'server_{i}_cost', '0')
                # ... (다른 서버 필드들도 유사하게 float/int 변환 및 유효성 검사)
                server_types_data.append({
                    'id': form_data.get(f'server_{i}_id', f'SrvType{i + 1}'),
                    'cost': float(cost_str),
                    'cpu_cores': int(form_data.get(f'server_{i}_cpu_cores', '0')),
                    'ram_gb': int(form_data.get(f'server_{i}_ram_gb', '0')),
                    'storage_tb': float(form_data.get(f'server_{i}_storage_tb', '0')),
                    'power_kva': float(form_data.get(f'server_{i}_power_kva', '0')),
                    'space_sqm': float(form_data.get(f'server_{i}_space_sqm', '0')),
                })
                if server_types_data[-1]['cost'] < 0: raise ValueError(
                    f"서버 유형 {server_types_data[-1]['id']}의 비용은 음수가 될 수 없습니다.")
                # 기타 필드에 대한 음수 및 유효성 검사 추가...

            service_demands_data = []
            for i in range(submitted_num_services):
                max_units_str = form_data.get(f'service_{i}_max_units')
                # ... (다른 서비스 필드들도 유사하게 float/int 변환 및 유효성 검사)
                service_demands_data.append({
                    'id': form_data.get(f'service_{i}_id', f'Svc{i + 1}'),
                    'revenue_per_unit': float(form_data.get(f'service_{i}_revenue_per_unit', '0')),
                    'req_cpu_cores': int(form_data.get(f'service_{i}_req_cpu_cores', '0')),
                    'req_ram_gb': int(form_data.get(f'service_{i}_req_ram_gb', '0')),
                    'req_storage_tb': float(form_data.get(f'service_{i}_req_storage_tb', '0')),
                    'max_units': int(max_units_str) if max_units_str and max_units_str.strip().isdigit() else None,
                })
                # 기타 필드에 대한 음수 및 유효성 검사 추가...

            logger.debug(f"Parsed Global Constraints: {global_constraints}")
            logger.debug(f"Parsed Server Types: {server_types_data}")
            logger.debug(f"Parsed Service Demands: {service_demands_data}")

            results_data, error_msg_opt, processing_time_ms = run_data_center_capacity_optimizer(global_constraints,
                                                                                                 server_types_data,
                                                                                                 service_demands_data)
            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                context['success_message'] = "데이터 센터 용량 계획 최적화 완료!"
                logger.info(
                    f"Data center capacity optimization successful. Total Profit: {results_data.get('total_profit')}")
            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다 (결과 없음)."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in data_center_capacity_demo_view (POST): {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in data_center_capacity_demo_view (POST): {e}", exc_info=True)

    return render(request, 'resource_allocation_app/data_center_capacity_demo.html', context)