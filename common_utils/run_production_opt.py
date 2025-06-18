from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
from ortools.sat.python import cp_model # CP-SAT 솔버 사용
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


def run_single_machine_optimizer(input_data):
    """
    OR-Tools CP-SAT를 사용하여 단일 기계 스케줄링 문제를 해결합니다.
    input_data: [{'id': 'A', 'processing_time': 10, 'due_date': 20}, ...]
    objective_choice: 최소화할 목표 (예: 'total_flow_time', 'total_tardiness')
    """
    objective_choice=input_data.get('objective_choice')
    logger.info(f"Running Single Machine Scheduler for objective: {objective_choice}")
    logger.debug(f"Jobs Data: {input_data}")

    num_jobs = input_data.get('num_jobs')
    if num_jobs == 0:
        return None, "오류: 작업 데이터가 없습니다.", 0.0

    model = cp_model.CpModel()

    jobs_list = input_data.get('jobs_list')
    # --- 1. 데이터 및 모델 범위(Horizon) 설정 ---
    all_processing_times = [j['processing_time'] for j in jobs_list]
    horizon = sum(all_processing_times)  # 모든 작업이 순차적으로 끝나는 시간

    # --- 2. 결정 변수 생성 ---
    # 각 작업의 시작 시간, 종료 시간, 기간(Interval) 변수
    start_vars = [model.NewIntVar(0, horizon, f'start_{i}') for i in range(num_jobs)]
    end_vars = [model.NewIntVar(0, horizon, f'end_{i}') for i in range(num_jobs)]
    interval_vars = [
        model.NewIntervalVar(start_vars[i], jobs_list[i]['processing_time'], end_vars[i], f'interval_{i}')
        for i in range(num_jobs)
    ]
    logger.debug(f"Created {num_jobs * 3} variables (start, end, interval). Horizon: {horizon}")

    # --- 3. 제약 조건 설정 ---
    # 3.1. No Overlap 제약: 단일 기계는 한 번에 하나의 작업만 처리
    model.AddNoOverlap(interval_vars)
    logger.debug("Added NoOverlap constraint.")

    # --- 4. 목표 함수 설정 ---
    if objective_choice == 'total_flow_time':
        # 총 흐름 시간(Total Completion Time) 최소화
        model.Minimize(sum(end_vars))
        logger.debug("Objective set to: Minimize Total Flow Time.")
    elif objective_choice == 'makespan':
        # 총 완료 시간(Makespan) 최소화
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)
        logger.debug("Objective set to: Minimize Makespan.")
    elif objective_choice == 'total_tardiness':
        # 총 지연 시간(Total Tardiness) 최소화
        tardiness_vars = [model.NewIntVar(0, horizon, f'tardiness_{i}') for i in range(num_jobs)]
        for i in range(num_jobs):
            due_date = input_data[i]['due_date']
            # T_i >= C_i - d_i
            model.Add(tardiness_vars[i] >= end_vars[i] - due_date)
        model.Minimize(sum(tardiness_vars))
        logger.debug("Objective set to: Minimize Total Tardiness.")
    else:
        # 기본 목표 또는 오류 처리
        logger.warning(f"Unknown objective '{objective_choice}'. Defaulting to total_flow_time.")
        model.Minimize(sum(end_vars))

    # --- 5. 문제 해결 ---
    solver = cp_model.CpSolver()
    logger.info("Solving the Single Machine Scheduling model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve(model)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {solver.StatusName(status)}, Time: {processing_time_ms:.2f} ms")

    # --- 6. 결과 추출 ---
    results = {'schedule': [], 'objective_value': 0}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results['objective_value'] = solver.ObjectiveValue()

        for i in range(num_jobs):
            results['schedule'].append({
                'id': jobs_list[i].get('id', f'Job {i + 1}'),
                'start': solver.Value(start_vars[i]),
                'end': solver.Value(end_vars[i]),
                'processing_time': jobs_list[i]['processing_time'],
                'due_date': jobs_list[i]['due_date']
            })

        # 시작 시간 순서로 결과 정렬
        results['schedule'].sort(key=lambda item: item['start'])

    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Single Machine Scheduling failed: {error_msg}")

    return results, error_msg, processing_time_ms
