import numpy as np
from django.conf import settings
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)

from common_utils.data_utils_allocation import *
import logging

from math import floor

logger = logging.getLogger('resource_allocation_app')


def run_budget_allocation_optimizer(input_data):
    total_budget = input_data.get('total_budget')
    items_data = input_data.get('items_data')
    logger.info(f"Running budget allocation for Total Budget: {total_budget}, Items: {len(items_data)}")

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        logger.error("GLOP Solver not available for budget allocation.")
        return None, 0, "오류: 선형 계획법 솔버(GLOP)를 생성할 수 없습니다.", 0.0

    num_items = input_data.get('num_items')
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
    status = solver.Solve()
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} ms")

    results = {'allocations': [],
               'total_maximized_return': 0,
               'total_allocated_budget': 0,
               'budget_utilization_percent':0}
    allocations = []
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found for budget allocation, but it might not be optimal.")
            # 사용자에게는 성공으로 알리고, 내부적으로만 경고
            # error_msg = "최적해일 수도 있지만, 더 좋은 해가 있을 가능성이 있습니다." # 이 메시지는 혼란을 줄 수 있어 제거

        raw_objective_value = solver.Objective().Value()
        total_maximized_return = raw_objective_value if raw_objective_value is not None else 0.0
        logger.info(f"Budget allocation objective value (Total Maximized Return): {total_maximized_return}")
        results['total_maximized_return'] = total_maximized_return
        calculated_total_allocated=0
        for i in range(num_items):
            allocated_val = x[i].solution_value()
            # 매우 작은 값은 0으로 처리 (부동소수점 정밀도 문제)
            if abs(allocated_val) < 1e-6: allocated_val = 0.0

            allocations.append({
                'name': items_data[i].get('name', f'항목 {i + 1}'),
                'allocated_budget': round(allocated_val, 2),
                'expected_return': round(allocated_val * float(items_data[i].get('return_coefficient', 0)), 2),
                'min_alloc': items_data[i].get('min_alloc'),
                'max_alloc': items_data[i].get('max_alloc'),
                'return_coefficient': items_data[i].get('return_coefficient')
            })
            calculated_total_allocated += round(allocated_val, 2)

        results['allocations'] = allocations
        results['total_allocated_budget'] = calculated_total_allocated
        if total_budget > 0:
            utilization_percent = (calculated_total_allocated / total_budget) * 100
            results['budget_utilization_percent'] = round(utilization_percent, 1)
        else:
            if calculated_total_allocated == 0:
                results['budget_utilization_percent'] = 0.0
            else:
                results['budget_utilization_percent'] = "N/A (Total Budget is 0)"
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
    return results, error_msg, get_solving_time_sec(solver.WallTime())


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


def run_datacenter_capacity_optimizer(input_data):


    logger.info("Running Data Center Capacity Optimizer...")
    global_constraints = input_data.get('global_constraints')
    server_data = input_data.get('server_data')
    demand_data = input_data.get('demand_data')
    logger.debug(f"Global Constraints: {global_constraints}")
    logger.debug(f"Server Data: {server_data}")
    logger.debug(f"Demands Data: {demand_data}")

    solver = pywraplp.Solver.CreateSolver('CBC')  # 또는 'SCIP' 등 MIP 지원 솔버
    if not solver:
        logger.error("CBC/SCIP MIP Solver not available for data center capacity planning.")
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    infinity = solver.infinity()
    num_server_data = input_data.get('num_server_types')
    num_demand_data = input_data.get('num_services')
    total_budget = global_constraints.get('total_budget')
    total_power = global_constraints.get('total_power_kva')
    total_space = global_constraints.get('total_space_sqm')

    # --- 결정 변수 ---
    # Sv[i]: 서버 i 구매 수(정수 변수)
    svr_name = [solver.IntVar(0, infinity, f'Sv{i+1}') for i in range(num_server_data)]
    logger.solve(f"SV: 서버 i의 구매 개수, 총 {len(svr_name)}개 생성")
    for i, var in enumerate(svr_name):
        ub = floor(min(total_power/server_data[i].get('power_kva'),total_space/server_data[i].get('space_sqm')))
        logger.solve(f"  - {var.name()} (서버: {server_data[i].get('id', i)}), 범위: [{var.lb()}, {ub}]")

     # Dm[s]Sv[i]: 서비스 s를 위해 서버 i에 할당된 "자원 단위" 또는 "서비스 인스턴스 수"
    # 여기서는, 각 서비스가 특정 양의 CPU, RAM, Storage를 요구하고,
    # 각 서버이 특정 양의 CPU, RAM, Storage를 제공한다고 가정.
    Alloc = {}
    for i_idx in range(num_server_data):
        for s_idx in range(num_demand_data):
            service = demand_data[s_idx]
            # 서비스 s의 최대 유닛 수 (수요) 만큼 변수 생성 고려
            # 또는, 총 제공 가능한 서비스 유닛을 변수로 할 수도 있음.
            # 여기서는 서비스 s를 서버 i에서 몇 '유닛'만큼 제공할지를 변수로 설정.
            # 이 '유닛'은 해당 서비스의 요구 자원에 맞춰짐.
            # 서비스 s를 서버 i에서 몇 유닛 제공할지 (이산적인 서비스 유닛으로 가정)
            max_units_s = service.get('max_units', infinity) if service.get('max_units') is not None else infinity
            Alloc[s_idx, i_idx] = solver.IntVar(0, max_units_s if max_units_s != infinity else solver.infinity(),
                                               f'Alloc{i_idx+1}_{s_idx+1}')

    logger.solve(f"Alloc_ij: 서버 i에 할당된 서비스 j의 용량, 총 {len(Alloc)}개 생성")
    # 모든 변수를 출력하기는 너무 많을 수 있으므로, 일부만 예시로 출력하거나 요약
    if len(Alloc) > 10:  # 변수가 많을 경우 일부만 출력
        logger.solve(
            f"  (예시) X_s{demand_data[0].get('id', 0)}_i{server_data[0].get('id', 0)}, X_s{demand_data[0].get('id', 0)}_i{server_data[1].get('id', 1)}, ...")
    else:
        for (s_idx, i_idx), var in Alloc.items():
            logger.solve(
                f"  - {var.name()} (서버: {server_data[i_idx].get('id', i_idx)}),서비스: {demand_data[s_idx].get('id', s_idx)},  범위: [{var.lb()}, {var.ub()}]")
    logger.solve(f"Created {len(svr_name)} Sv variables and {len(Alloc)} Alloc variables.")

    # --- 제약 조건 ---
    logger.solve("**제약 조건:**")

    # 1. 총 예산, 전력, 공간 제약
    total_budget_constraint = solver.Constraint(0, global_constraints.get('total_budget', infinity), 'total_budget')
    total_power_constraint = solver.Constraint(0, global_constraints.get('total_power_kva', infinity), 'total_power')
    total_space_constraint = solver.Constraint(0, global_constraints.get('total_space_sqm', infinity), 'total_space')
    for i in range(num_server_data):
        total_budget_constraint.SetCoefficient(svr_name[i], server_data[i].get('cost', 0))
        total_power_constraint.SetCoefficient(svr_name[i], server_data[i].get('power_kva', 0))
        total_space_constraint.SetCoefficient(svr_name[i], server_data[i].get('space_sqm', 0))

    budget_terms= []
    power_terms= []
    space_terms = []
    for i in range(num_server_data):
        cost = server_data[i].get('cost', 0)
        power = server_data[i].get('power_kva', 0)
        space = server_data[i].get('space_sqm', 0)
        if cost != 0:
            budget_terms.append(f"{cost}*{svr_name[i].name()}")
        if power != 0:
            power_terms.append(f"{power}*{svr_name[i].name()}")
        if space != 0:
            space_terms.append(f"{space}*{svr_name[i].name()}")
    budget_expr_str = " + ".join(budget_terms)
    power_expr_str = " + ".join(power_terms)
    space_expr_str = " + ".join(space_terms)
    logger.solve(
        f"total_budget: {total_budget_constraint.lb()} <= {budget_expr_str} <= {total_budget_constraint.ub()}")
    logger.solve(
        f"total_power: {total_power_constraint.lb()} <= {power_expr_str} <= {total_power_constraint.ub()}")
    logger.solve(
        f"total_space: {total_space_constraint.lb()} <= {space_expr_str} <= {total_space_constraint.ub()}")

    # 4. 각 자원(CPU, RAM, Storage)에 대한 용량 제약
    # 각 서버 i가 제공하는 총 CPU = Ns[i] * server_data[i]['cpu_cores']
    # 각 서비스 s의 유닛이 요구하는 CPU = demands_data[s]['req_cpu_cores']
    # 총 요구 CPU = sum over s,i (X_si[s,i] * demands_data[s]['req_cpu_cores'])
    # 이는 잘못된 접근. X_si는 서비스 s를 서버 i에서 몇 유닛 제공하는지.
    # 서버 i에 할당된 서비스들의 총 요구 자원이 서버 i의 총 제공 자원을 넘을 수 없음.

    # 수정된 제약: 각 서버 i에 대해, 해당 서버에 할당된 모든 서비스의 자원 요구량 합계는
    # 해당 서버의 총 구매된 용량을 초과할 수 없음.
    resource_types = ['cpu_cores', 'ram_gb', 'storage_tb']
    for i_idx in range(num_server_data):  # 각 서버에 대해
        server_type = server_data[i_idx]
        for res_idx, resource in enumerate(resource_types):  # 각 자원 유형에 대해
            # # 서버 i가 제공하는 총 자원량
            # # Ns[i_idx] * server.get(resource, 0)
            # # 서버 i에 할당된 모든 서비스 유닛들이 소모하는 총 자원량
            # # sum (X_si[s_idx, i_idx] * demands_data[s_idx].get(f'req_{resource}', 0) for s_idx in range(num_services))
            # constraint_res = solver.Constraint(-infinity, 0, f'res_{resource}server{i_idx}')
            # # 제공량 (우변으로 넘기면 <= 0)
            # constraint_res.SetCoefficient(Ns[i_idx], -server.get(resource, 0))  # 제공량은 음수로
            # # 소비량 (좌변에 그대로)
            # for s_idx in range(num_services):
            #     if (s_idx, i_idx) in X_si:  # 해당 변수가 존재할 때만
            #         service = demands_data[s_idx]
            #         constraint_res.SetCoefficient(X_si[s_idx, i_idx], service.get(f'req_{resource}', 0))  # 소비량은 양수로
            # logger.solve(f"Added resource constraint for {resource} on server type {server.get('id', i_idx)}.")

            # 제약 조건을 생성하기 전에 정의된 변수가 존재하는지 확인
            coeffs = []
            terms = []
            # Ns[i_idx] * server.get(resource, 0) 만큼의 자원 제공
            coeffs.append(-server_type.get(resource, 0))
            terms.append(svr_name[i_idx])

            for s_idx in range(num_demand_data):
                if (s_idx, i_idx) in Alloc:
                    service = demand_data[s_idx]
                    req_resource = service.get(f'req_{resource}', 0)
                    coeffs.append(req_resource)
                    terms.append(Alloc[s_idx, i_idx])

            # 모든 계수가 0이 아닌 경우에만 제약을 추가 (자원 요구사항이 없는 경우 불필요)
            if any(c != 0 for c in coeffs):
                constraint_expr = solver.Sum(terms[j] * coeffs[j] for j in range(len(terms)))
                constraint_name = f'con_{resource}_{server_type.get("id", i_idx)}'
                # sum(X_si[s,i] * req_res[s]) <= Ns[i] * server_res[i] 형태로 표현 가능
                # 즉, sum(X_si[s,i] * req_res[s]) - Ns[i] * server_res[i] <= 0
                constraint = solver.Add(constraint_expr <= 0, constraint_name)
                logger.solve(f"{constraint.name()}: {constraint_expr} <= 0")

    # 5. 각 서비스의 최대 수요(유닛) 제약 (선택 사항, X_si 변수 상한으로 이미 반영됨)
    # sum over i (X_si[s,i]) <= demands_data[s]['max_units'] (또는 == nếu 정확히 수요 충족)
    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        max_units_s = service.get('max_units')
        if max_units_s is not None and max_units_s != infinity:
            # 서비스 s에 대해 모든 서버에서 제공되는 총 유닛 수는 max_units_s를 넘을 수 없음
            constraint_demand_s = solver.Constraint(0, max_units_s, f'demand_service_{s_idx}')
            for i_idx in range(num_server_data):
                if (s_idx, i_idx) in Alloc:
                    constraint_demand_s.SetCoefficient(Alloc[s_idx, i_idx], 1)
            # logger.solve(f"service_{service.get('id', s_idx)}: sum(X_si[{service.get('id', s_idx)},i]) <= {max_units_s}")
            logger.solve(f"service_{service.get('id', s_idx)}: sum(Dm[{s_idx},i]) <= {max_units_s}")
    # --- 목표 함수 ---
    # 총 이익 = (각 서비스 유닛 수익 합계) - (총 서버 구매 비용)
    objective = solver.Objective()
    # 서버 구매 비용 (음수)
    for i in range(num_server_data):
        objective.SetCoefficient(svr_name[i], -server_data[i].get('cost', 0))

    # 서비스 수익 (양수)
    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        for i_idx in range(num_server_data):
            if (s_idx, i_idx) in Alloc:
                objective.SetCoefficient(Alloc[s_idx, i_idx], service.get('revenue_per_unit', 0))

    objective.SetMaximization()
    logger.solve(f"\n**목표 함수:** 총 이익 극대화 (서비스 수익 - 서버 구매 비용)")
    logger.solve(f"  목표: Maximize sum(X_si * revenue_per_unit) - sum(Ns * cost)")

    obj_terms = []
    for i in range(num_server_data):
        coff = server_data[i].get('cost', 0)
        if coff != 0:
            obj_terms.append(f"{-coff}*{svr_name[i].name()}")

    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        for i_idx in range(num_server_data):
            if (s_idx, i_idx) in Alloc:
                objective.SetCoefficient(Alloc[s_idx, i_idx], service.get('revenue_per_unit', 0))
                coff = service.get('revenue_per_unit',0)
                if coff != 0:
                    obj_terms.append(f"{coff}*{Alloc[s_idx, i_idx].name()}")
    obj_expr_str = " + ".join(obj_terms)
    logger.solve(f"obj_exp: {obj_expr_str}")
    # --- 문제 해결 ---
    logger.info("Solving Data Center Capacity model...")
    status = solver.Solve()
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} ms")

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
        for i in range(num_server_data):
            num_purchased = svr_name[i].solution_value()
            if abs(num_purchased) < 1e-6: num_purchased = 0  # 부동소수점 정리
            num_purchased = int(round(num_purchased))  # 정수 변수이므로 반올림

            if num_purchased > 0:
                server_type = server_data[i]
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
        for s_idx in range(num_demand_data):
            service = demand_data[s_idx]
            total_units_for_service_s = 0
            allocation_details_s = []
            for i_idx in range(num_server_data):
                if (s_idx, i_idx) in Alloc:
                    units_on_server_i = Alloc[s_idx, i_idx].solution_value()
                    if abs(units_on_server_i) < 1e-6: units_on_server_i = 0
                    units_on_server_i = int(round(units_on_server_i))

                    if units_on_server_i > 0:
                        total_units_for_service_s += units_on_server_i
                        allocation_details_s.append({
                            'server_type_id': server_data[i_idx].get('id', f'Type{i_idx}'),
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

    return results, error_msg, get_solving_time_sec(solver.WallTime())


def run_nurse_rostering_optimizer(input_data):
    """
    주말 근무 공정성, 최소 휴식 보장 등 복잡한 제약이 추가된
    간호사 스케줄링 문제를 해결합니다.
    """
    # 입력 데이터 파싱
    num_nurses = input_data['num_nurses']
    num_days = input_data['num_days']
    shifts = input_data['shifts']
    day_shift_requests = input_data['day_shift_requests']
    min_shifts_per_nurse = input_data['min_shifts_per_nurse']
    max_shifts_per_nurse = input_data['max_shifts_per_nurse']
    weekend_days = input_data['weekend_days']  # 주말에 해당하는 날짜 인덱스
    logger.info(
        f"Running Nurse Rostering Optimizer. - Num nurses: {num_nurses},  Num days: {num_days}, Shifts: {shifts}")
    all_nurses = range(num_nurses)
    all_days = range(num_days)
    all_shifts = range(len(shifts))
    num_shifts_per_day = len(shifts)

    try:
        model = cp_model.CpModel()

        # --- 1. 결정 변수 생성 ---
        assigns = {}
        for n in all_nurses:
            for d in all_days:
                for s in all_shifts:
                    assigns[(n, d, s)] = model.NewBoolVar(f'assigns_n{n}_d{d}_s{s}')

        # --- 2. 강성 제약 조건 (Hard Constraints) ---

        # 제약 1: 각 간호사는 하루에 최대 하나의 시프트만 근무
        for n in all_nurses:
            for d in all_days:
                model.AddAtMostOne(assigns[(n, d, s)] for s in all_shifts)

        # 제약 2: 각 날짜, 각 시프트의 필요 인원 충족
        for d in all_days:
            for s in all_shifts:
                required_nurses = day_shift_requests[d][s]
                model.Add(sum(assigns[(n, d, s)] for n in all_nurses) == required_nurses)

        # 제약 3: [신규] 최소 휴식 보장 (근무 후 2개 시프트 휴식)
        for n in all_nurses:
            for d in range(num_days - 1):
                for s in range(num_shifts_per_day):
                    # 현재 시프트의 인덱스를 전체 시간 흐름으로 변환
                    current_shift_index = d * num_shifts_per_day + s

                    # 다음 2개 시프트에 대한 근무 금지 제약 추가
                    for i in range(1, 3):
                        next_shift_index = current_shift_index + i
                        next_d = next_shift_index // num_shifts_per_day
                        next_s = next_shift_index % num_shifts_per_day

                        if next_d < num_days:
                            # "간호사 n이 (d,s)에 근무하면, (next_d, next_s)에는 근무할 수 없다"
                            model.AddImplication(assigns[(n, d, s)], assigns[(n, next_d, next_s)].Not())

        # --- 3. 연성 제약 조건 (Soft Constraints) ---

        # 목표 1: 근무일 수가 min/max 범위에서 벗어나는 정도 최소화
        total_shifts_worked = [sum(assigns[(n, d, s)] for d in all_days for s in all_shifts) for n in all_nurses]
        penalties = []
        for n in all_nurses:
            under_penalty = model.NewIntVar(0, num_days, f'under_penalty_n{n}')
            over_penalty = model.NewIntVar(0, num_days, f'over_penalty_n{n}')
            model.Add(min_shifts_per_nurse - total_shifts_worked[n] <= under_penalty)
            model.Add(total_shifts_worked[n] - max_shifts_per_nurse <= over_penalty)
            penalties.append(under_penalty)
            penalties.append(over_penalty)

        # 목표 2: [신규] 주말 근무 공평 분배
        weekend_shifts_worked = [sum(assigns[(n, d, s)] for d in weekend_days for s in all_shifts) for n in all_nurses]

        min_weekend_shifts = model.NewIntVar(0, num_days, 'min_weekend_shifts')
        max_weekend_shifts = model.NewIntVar(0, num_days, 'max_weekend_shifts')
        model.AddMinEquality(min_weekend_shifts, weekend_shifts_worked)
        model.AddMaxEquality(max_weekend_shifts, weekend_shifts_worked)
        weekend_gap = max_weekend_shifts - min_weekend_shifts

        # --- 4. 목표 함수 설정 ---
        # 페널티 총합 + 주말 근무 격차 최소화 (주말 근무 공정성에 더 높은 가중치 부여)
        model.Minimize(sum(penalties) + weekend_gap * 2)

        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = settings.ORTOOLS_TIME_LIMIT
        status = solver.Solve(model)
        processing_time = solver.WallTime()
        logger.info(f"Solver status: {status}, Time: {processing_time} ms")

        # --- 6. 결과 추출 ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = {}
            for d in all_days:
                schedule[d] = {}
                for s in all_shifts:
                    schedule[d][s] = [n for n in all_nurses if solver.Value(assigns[(n, d, s)]) == 1]

            results_data = {
                'schedule': schedule,
                'nurse_work_days': [solver.Value(s) for s in total_shifts_worked],
                'weekend_work_days': [solver.Value(w) for w in weekend_shifts_worked],
                'total_penalty': solver.ObjectiveValue()
            }
            return results_data, None, processing_time
        else:
            return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격할 수 있습니다.", processing_time

    except Exception as e:
        return None, f"오류 발생: {str(e)}", None


def run_nurse_roster_advanced_optimizer(input_data):
    """
    숙련도, 휴가, 강화된 공정성 등 고급 제약이 포함된 스케줄링 문제를 해결합니다.
    """
    SHIFT_NIGHT = preset_nurse_rostering_shifts[2]
    # --- 입력 데이터 파싱 ---
    nurses_data = input_data['nurses_data']
    num_nurses = len(nurses_data)
    num_days = input_data['num_days']
    shifts = input_data['shifts']

    # 숙련도별 필요 인원
    skill_requirements = input_data['skill_requirements']
    all_skills = list(skill_requirements[shifts[0]].keys())  # ['상', '중', '하']

    # 간호사별 휴가 요청
    vacation_requests = input_data['vacation_requests']  # {nurse_id: [day1, day2], ...}

    # 선택적으로 적용할 공정성 제약
    enabled_fairness = input_data.get('enabled_fairness', [])

    weekend_days = input_data['weekend_days']
    num_shifts_per_day = len(shifts)

    # 간호사 ID와 인덱스, 스킬 매핑
    nurse_ids = [n['id'] for n in nurses_data]
    nurses_by_skill = {skill: [n['id'] for n in nurses_data if n['skill'] == skill] for skill in all_skills}
    logger.info(
        f"Running Nurse Rostering Advanced Optimizer. - Num nurses: {num_nurses},  Num days: {num_days}, Shifts: {shifts}")

    try:
        model = cp_model.CpModel()

        # --- 1. 결정 변수 생성 ---
        assigns = {}
        for n_id in nurse_ids:
            for d in range(num_days):
                for s in range(num_shifts_per_day):
                    assigns[(n_id, d, s)] = model.NewBoolVar(f'assigns_n{n_id}_d{d}_s{s}')

        # --- 2. 강성 제약 조건 (Hard Constraints) ---

        # 제약 1: 각 간호사는 하루 최대 1개 시프트 근무
        for n_id in nurse_ids:
            for d in range(num_days):
                model.AddAtMostOne(assigns[(n_id, d, s)] for s in range(num_shifts_per_day))

        # 제약 2: [수정] 숙련도별 필요 인원 충족
        for d in range(num_days):
            for s_idx, s_name in enumerate(shifts):
                for skill, required_count in skill_requirements[s_name].items():
                    nurses_with_that_skill = nurses_by_skill[skill]
                    model.Add(sum(assigns[(n_id, d, s_idx)] for n_id in nurses_with_that_skill) >= required_count)

        # 제약 3: [신규] 휴가 요청 반영
        for n_id, off_days in vacation_requests.items():
            for d in off_days:
                model.Add(sum(assigns[(n_id, d, s)] for s in range(num_shifts_per_day)) == 0)

        # --- 3. 연성 제약 조건 (Soft Constraints) 및 목표 함수 ---

        # 목표 1: [신규] 공평한 야간 근무 분배
        if 'fair_nights' in enabled_fairness:
            night_shift_idx = shifts.index(SHIFT_NIGHT)
            night_shifts_worked = [sum(assigns[(n_id, d, night_shift_idx)] for d in range(num_days)) for n_id in
                                   nurse_ids]
            min_nights = model.NewIntVar(0, num_days, 'min_nights')
            max_nights = model.NewIntVar(0, num_days, 'max_nights')
            model.AddMinEquality(min_nights, night_shifts_worked)
            model.AddMaxEquality(max_nights, night_shifts_worked)
            night_gap = max_nights - min_nights
        else:
            night_gap = 0

        # 목표 2: [신규] 공평한 휴무일 분배
        if 'fair_offs' in enabled_fairness:
            total_shifts_worked = [
                sum(assigns[(n_id, d, s)] for d in range(num_days) for s in range(num_shifts_per_day)) for n_id in
                nurse_ids]
            off_days_per_nurse = [num_days - s for s in total_shifts_worked]
            min_offs = model.NewIntVar(0, num_days, 'min_offs')
            max_offs = model.NewIntVar(0, num_days, 'max_offs')
            model.AddMinEquality(min_offs, off_days_per_nurse)
            model.AddMaxEquality(max_offs, off_days_per_nurse)
            off_gap = max_offs - min_offs
        else:
            off_gap = 0

        # 목표 3: [기존] 공평한 주말 근무 분배
        if 'fair_weekends' in enabled_fairness:
            weekend_shifts_worked = [sum(assigns[(n_id, d, s)] for d in weekend_days for s in range(num_shifts_per_day))
                                     for n_id in nurse_ids]
            min_weekend_shifts = model.NewIntVar(0, len(weekend_days), 'min_weekend')
            max_weekend_shifts = model.NewIntVar(0, len(weekend_days), 'max_weekend')
            model.AddMinEquality(min_weekend_shifts, weekend_shifts_worked)
            model.AddMaxEquality(max_weekend_shifts, weekend_shifts_worked)
            weekend_gap = max_weekend_shifts - min_weekend_shifts
        else:
            weekend_gap = 0
            weekend_shifts_worked = [0] * num_nurses  # 결과 표시를 위한 기본값

        # --- 4. 목표 함수 설정 ---
        # 각 공정성 목표의 격차(gap) 합을 최소화
        model.Minimize(night_gap * 2 + off_gap + weekend_gap * 3)  # 야간, 주말에 가중치 부여

        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        processing_time = solver.WallTime()
        logger.info(f"Solver status: {status}, Time: {processing_time} ms")

        # --- 6. 결과 추출 ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = {}
            for d in range(num_days):
                schedule[d] = {}
                for s_idx, s_name in enumerate(shifts):
                    schedule[d][s_idx] = [n_id for n_id in nurse_ids if solver.Value(assigns[(n_id, d, s_idx)]) == 1]

            # 각 간호사별 통계 계산
            total_shifts = [
                sum(solver.Value(assigns[(n_id, d, s)]) for d in range(num_days) for s in range(num_shifts_per_day)) for
                n_id in nurse_ids]
            if 'fair_nights' in enabled_fairness and SHIFT_NIGHT in shifts:
                night_shift_idx = shifts.index(SHIFT_NIGHT)
                total_nights = [sum(solver.Value(assigns[(n_id, d, night_shift_idx)]) for d in range(num_days)) for n_id
                                in nurse_ids]
            else:
                total_nights = [0] * num_nurses
            total_weekends = [solver.Value(w) for w in
                              weekend_shifts_worked] if 'fair_weekends' in enabled_fairness else [0] * num_nurses
            total_offs = [num_days - ts for ts in total_shifts]

            results_data = {
                'schedule': schedule,
                'nurse_stats': {
                    n_id: {
                        'total': total_shifts[i], 'nights': total_nights[i],
                        'weekends': total_weekends[i], 'offs': total_offs[i]
                    } for i, n_id in enumerate(nurse_ids)
                },
                'total_penalty': solver.ObjectiveValue()
            }
            logger.info(f'results_data:{results_data}')
            return results_data, None, processing_time
        else:
            return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", round(processing_time, 4)

    except Exception as e:
        return None, f"오류 발생: {str(e)}", None

def get_solving_time_sec(processing_time):
    # solver.WallTime(): if solver is CP-SAT then, sec else ms
    processing_time = processing_time / 1000
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"