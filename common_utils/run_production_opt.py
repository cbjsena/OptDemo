from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
import datetime
import logging

logger = logging.getLogger(__name__)

# --- Lot Sizing 최적화 실행 함수 ---
def run_lot_sizing_optimizer(input_data):
    """
    OR-Tools를 사용하여 Lot Sizing 문제를 해결합니다.
    """
    logger.info("Running Lot Sizing Optimizer.")

    demands = input_data.get('demands', [])
    num_periods = len(demands)
    if num_periods == 0:
        return None, "오류: 수요 데이터가 없습니다.", 0.0

    setup_costs = input_data.get('setup_costs')
    prod_costs = input_data.get('prod_costs')
    holding_costs = input_data.get('holding_costs')
    capacities = input_data.get('capacities')

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    infinity = solver.infinity()

    # 변수 생성
    x = [solver.IntVar(0, infinity, f'x_{t}') for t in range(num_periods)]  # 생산량
    y = [solver.BoolVar(f'y_{t}') for t in range(num_periods)]  # 생산 여부
    I = [solver.IntVar(0, infinity, f'I_{t}') for t in range(num_periods)]  # 재고량

    # 제약 조건
    # 1. 재고 균형 제약
    for t in range(num_periods):
        previous_inventory = I[t - 1] if t > 0 else 0
        solver.Add(previous_inventory + x[t] == demands[t] + I[t], f"inventory_balance_{t}")

    # 2. 생산 용량 제약
    for t in range(num_periods):
        solver.Add(x[t] <= capacities[t] * y[t], f"capacity_constraint_{t}")

    # 목표 함수: 총비용 최소화
    objective = solver.Objective()
    for t in range(num_periods):
        objective.SetCoefficient(y[t], setup_costs[t])
        objective.SetCoefficient(x[t], prod_costs[t])
        objective.SetCoefficient(I[t], holding_costs[t])
    objective.SetMinimization()

    logger.info("Solving the Lot Sizing model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time_ms:.2f} ms")

    # 결과 추출
    results = {'schedule': [], 'total_cost': 0}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        results['total_cost'] = solver.Objective().Value()
        total_setup_cost = 0
        total_prod_cost = 0
        total_holding_cost = 0

        for t in range(num_periods):
            results['schedule'].append({
                'period': t + 1,
                'demand': demands[t],
                'production_amount': round(x[t].solution_value()),
                'inventory_level': round(I[t].solution_value()),
                'is_setup': bool(y[t].solution_value())
            })
            if y[t].solution_value() > 0.5:
                total_setup_cost += setup_costs[t]
            total_prod_cost += prod_costs[t] * x[t].solution_value()
            total_holding_cost += holding_costs[t] * I[t].solution_value()

        results['cost_breakdown'] = {
            'setup': round(total_setup_cost, 2),
            'production': round(total_prod_cost, 2),
            'holding': round(total_holding_cost, 2),
        }

    else:
        if status == pywraplp.Solver.INFEASIBLE:
            error_msg = "실행 불가능한 문제입니다. 모든 기간의 수요를 생산 능력 내에서 만족시킬 수 있는지 확인하세요."
        else:
            error_msg = f"최적해를 찾지 못했습니다. (솔버 상태: {status})"
        logger.error(f"Lot Sizing optimization failed: {error_msg}")

    return results, error_msg, processing_time_ms