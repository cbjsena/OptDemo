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


def run_flow_shop_optimizer(input_data):
    logger.info("Running Flow Shop Optimizer.")

    processing_times = input_data['processing_times']
    num_jobs = input_data['num_jobs']
    num_machines = input_data['num_machines']

    model = cp_model.CpModel()

    # Horizon 계산
    horizon = sum(sum(job) for job in processing_times)

    # 변수 생성: C_ij (작업 i가 기계 j에서 끝나는 시간)
    completion_times = [[model.NewIntVar(0, horizon, f'C_{i}_{j}') for j in range(num_machines)] for i in
                        range(num_jobs)]

    # 제약 조건
    # 1. 기계 순서 제약 (작업 흐름)
    for i in range(num_jobs):
        for j in range(1, num_machines):
            model.Add(completion_times[i][j] >= completion_times[i][j - 1] + processing_times[i][j])

    # 2. 작업 순서 제약 (기계 독점) - 순열(Permutation) 플로우샵 가정
    # y_ik = 1 if job i is before job k
    y = {(i, k): model.NewBoolVar(f'y_{i}_{k}') for i in range(num_jobs) for k in range(num_jobs) if i < k}

    for j in range(num_machines):  # 모든 기계에서
        for i in range(num_jobs):
            for k in range(i + 1, num_jobs):
                # 작업 i가 k보다 먼저 끝나거나, k가 i보다 먼저 끝나야 함
                # C_kj >= C_ij + p_kj OR C_ij >= C_kj + p_ij
                # BigM 기법 사용
                # C_kj - (C_ij + p_ij) >= 0 또는 C_ij - (C_kj + p_kj) >= 0
                # 이 부분을 CP-SAT의 AddNoOverlap으로 더 효율적으로 모델링 가능
                pass  # 아래 NoOverlap으로 대체

    # 2. (개선된) 작업 순서 제약: NoOverlap 사용
    for j in range(num_machines):
        intervals = []
        for i in range(num_jobs):
            start_var = model.NewIntVar(0, horizon, f'start_{i}_{j}')
            # C_ij = start_ij + p_ij
            model.Add(completion_times[i][j] == start_var + processing_times[i][j])
            intervals.append(
                model.NewIntervalVar(start_var, processing_times[i][j], completion_times[i][j], f'interval_{i}_{j}'))
        model.AddNoOverlap(intervals)

    # 목표 함수: Makespan 최소화
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [completion_times[i][num_machines - 1] for i in range(num_jobs)])
    model.Minimize(makespan)

    # 해결
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0  # 시간 제한
    solve_start_time = datetime.datetime.now()
    status = solver.Solve(model)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000

    # 결과 추출
    results = {'schedule': [], 'makespan': 0, 'sequence': []}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # 최적 순서 결정
        sequence_info = []
        for i in range(num_jobs):
            start_time_on_m0 = solver.Value(completion_times[i][0]) - processing_times[i][0]
            sequence_info.append({'job_index': i, 'start_time': start_time_on_m0})
        sequence_info.sort(key=lambda item: item['start_time'])

        optimal_sequence_indices = [item['job_index'] for item in sequence_info]
        optimal_sequence_ids = [input_data['job_ids'][i] for i in optimal_sequence_indices]

        # 계산된 최적 순서로 스케줄 및 Makespan 재계산 (결과 일관성 및 재사용성)
        results = calculate_flow_shop_schedule(
            processing_times,
            input_data['job_ids'],
            optimal_sequence_ids
        )
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"

    return results, error_msg, processing_time_ms


# --- 고정된 순서에 대한 Makespan 계산 함수 (새로 추가) ---
def calculate_flow_shop_schedule(processing_times, job_ids, sequence):
    """
    주어진 작업 순서(sequence)에 따라 Flow Shop 스케줄과 Makespan을 계산합니다.
    processing_times: [[p_ij, ...], ...]
    job_ids: ['Job 1', 'Job 2', ...]
    sequence: 순서를 나타내는 job_id 리스트. 예: ['Job 2', 'Job 1', 'Job 3']
    """
    num_jobs = len(processing_times)
    num_machines = len(processing_times[0]) if num_jobs > 0 else 0

    # job_id를 인덱스로 변환
    job_id_to_index = {job_id: i for i, job_id in enumerate(job_ids)}
    try:
        sequence_indices = [job_id_to_index[job_id] for job_id in sequence]
    except KeyError as e:
        raise ValueError(f"잘못된 작업 ID가 수동 순서에 포함되어 있습니다: {e}")

    if len(sequence_indices) != num_jobs or len(set(sequence_indices)) != num_jobs:
        raise ValueError("수동 순서에는 모든 작업이 정확히 한 번씩 포함되어야 합니다.")

    # 완료 시간 행렬 C_ij 초기화
    completion_times = [[0] * num_machines for _ in range(num_jobs)]

    # 재귀적 관계를 사용하여 완료 시간 계산
    for k in range(num_jobs):  # 순서 k (0 to n-1)
        job_idx = sequence_indices[k]
        for j in range(num_machines):  # 기계 j (0 to m-1)
            # 첫 번째 작업(k=0) 또는 첫 번째 기계(j=0)의 완료 시간
            prev_job_completion_on_same_machine = completion_times[sequence_indices[k - 1]][j] if k > 0 else 0
            prev_machine_completion_for_same_job = completion_times[job_idx][j - 1] if j > 0 else 0

            completion_times[job_idx][j] = max(prev_job_completion_on_same_machine,
                                               prev_machine_completion_for_same_job) + processing_times[job_idx][j]

    # Makespan은 마지막 순서의 작업이 마지막 기계에서 끝나는 시간
    makespan = completion_times[sequence_indices[-1]][num_machines - 1]

    # 간트 차트용 데이터 생성
    schedule = []
    for i in range(num_jobs):
        job_schedule = {'job_id': job_ids[i], 'tasks': []}
        for j in range(num_machines):
            end_time = completion_times[i][j]
            start_time = end_time - processing_times[i][j]
            job_schedule['tasks'].append({
                'machine': f'Machine {j + 1}',
                'start': start_time,
                'duration': processing_times[i][j],
                'end': end_time
            })
        schedule.append(job_schedule)
    results = {'schedule': schedule, 'makespan': makespan, 'sequence': sequence}

    return results


# --- Job Shop 최적화 실행 함수 ---
def run_job_shop_optimizer(input_data):
    logger.info("Running Job Shop Optimizer.")
    jobs_data = input_data['jobs']
    num_jobs = input_data['num_jobs']
    num_machines = input_data['num_machines']

    model = cp_model.CpModel()
    horizon = sum(task[1] for job in jobs_data for task in job)

    # 변수 생성: task (job_id, op_id)의 start, end, interval
    all_tasks = {}
    for i, job in enumerate(jobs_data):
        for j, task in enumerate(job):
            start_var = model.NewIntVar(0, horizon, f'start_{i}_{j}')
            end_var = model.NewIntVar(0, horizon, f'end_{i}_{j}')
            interval_var = model.NewIntervalVar(start_var, task[1], end_var, f'interval_{i}_{j}')
            all_tasks[(i, j)] = interval_var

    # 제약 조건
    # 1. 기계 독점 제약 (No Overlap)
    for j in range(num_machines):
        intervals_on_machine = []
        for i in range(num_jobs):
            for k in range(num_machines):
                if jobs_data[i][k][0] == j:
                    intervals_on_machine.append(all_tasks[(i, k)])
        model.AddNoOverlap(intervals_on_machine)

    # 2. 작업 내 공정 순서 제약 (Precedence)
    for i in range(num_jobs):
        for j in range(num_machines - 1):
            model.Add(all_tasks[(i, j + 1)].StartExpr() >= all_tasks[(i, j)].EndExpr())

    # 목표 함수: Makespan 최소화
    makespan = model.NewIntVar(0, horizon, 'makespan')
    all_end_times = [all_tasks[(i, num_machines - 1)].EndExpr() for i in range(num_jobs)]
    model.AddMaxEquality(makespan, all_end_times)
    model.Minimize(makespan)

    # 해결
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    processing_time_ms = solver.WallTime() * 1000

    # 결과 추출
    results = {'schedule': [], 'makespan': 0}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results['makespan'] = solver.ObjectiveValue()

        job_schedules = []
        for i in range(num_jobs):
            job_schedule = {'job_id': f'Job {i + 1}', 'tasks': []}
            for j in range(num_machines):
                machine_id = jobs_data[i][j][0]
                start_time = solver.Value(all_tasks[(i, j)].StartExpr())
                end_time = solver.Value(all_tasks[(i, j)].EndExpr())
                job_schedule['tasks'].append({
                    'machine': f'Machine {machine_id + 1}',
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
            job_schedules.append(job_schedule)
        results['schedule'] = job_schedules
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"

    return results, error_msg, processing_time_ms