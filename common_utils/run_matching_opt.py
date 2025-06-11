from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
from ortools.graph.python import linear_sum_assignment # 할당 문제 전용 솔버
import datetime
import logging

logger = logging.getLogger(__name__)


def run_matching_cf_tft_algorithm(cf_panels, tft_panels):
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)

    logger.info(f"Starting matching algorithm for {num_cf} CF panel(s) and {num_tft} TFT panel(s).")

    if num_cf == 0 or num_tft == 0:
        msg = "Matching algorithm called with zero CF or TFT panels."
        logger.warning(msg)
        return [], 0, msg

    # --- 1. 수율 매트릭스 (C_ij) 계산 ---
    yield_matrix = [[-1] * num_tft for _ in range(num_cf)]

    logger.debug("Calculating yield matrix...")
    for i in range(num_cf):
        cf_panel = cf_panels[i]
        cf_map = cf_panel.get('defect_map')
        cf_rows = cf_panel.get('rows')
        cf_cols = cf_panel.get('cols')

        # 필수 키 누락 또는 유효하지 않은 값(None 또는 0) 확인
        if not all([isinstance(cf_map, list), isinstance(cf_rows, int), cf_rows > 0, isinstance(cf_cols, int),
                    cf_cols > 0]):
            logger.warning(f"CF Panel {cf_panel.get('id', i)} has invalid structure or dimensions. Skipping.")
            continue  # 이 CF 패널은 모든 TFT와 매칭 불가 (-1 유지)

        for j in range(num_tft):
            tft_panel = tft_panels[j]
            tft_map = tft_panel.get('defect_map')
            tft_rows = tft_panel.get('rows')
            tft_cols = tft_panel.get('cols')

            if not all([isinstance(tft_map, list), isinstance(tft_rows, int), tft_rows > 0, isinstance(tft_cols, int),
                        tft_cols > 0]):
                logger.warning(
                    f"TFT Panel {tft_panel.get('id', j)} has invalid structure or dimensions. Marking as unmatchable with CF {cf_panel.get('id', i)}.")
                # yield_matrix[i][j]는 이미 -1
                continue

            if cf_rows != tft_rows or cf_cols != tft_cols:
                logger.debug(
                    f"Dimension mismatch between CF {cf_panel.get('id', i)} ({cf_rows}x{cf_cols}) and TFT {tft_panel.get('id', j)} ({tft_rows}x{tft_cols}).")
                # yield_matrix[i][j]는 이미 -1
                continue

            current_yield = 0
            valid_cell_structure = True
            if len(cf_map) != cf_rows or len(tft_map) != tft_rows:  # defect_map의 행 개수 확인
                logger.warning(
                    f"Defect map row count mismatch for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                valid_cell_structure = False

            if valid_cell_structure:
                for r in range(cf_rows):
                    if not valid_cell_structure: break
                    # 각 행의 열 개수 및 셀 값 유효성 확인
                    if len(cf_map[r]) != cf_cols or len(tft_map[r]) != tft_cols:
                        logger.warning(
                            f"Defect map col count mismatch at row {r} for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                        valid_cell_structure = False
                        break

                    for c in range(cf_cols):
                        cf_cell = cf_map[r][c]
                        tft_cell = tft_map[r][c]
                        if not (cf_cell in (0, 1) and tft_cell in (0, 1)):
                            logger.warning(
                                f"Invalid cell value at ({r},{c}) for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                            valid_cell_structure = False
                            break

                        if cf_cell == 0 and tft_cell == 0:  # 양품 조건
                            current_yield += 1
                    if not valid_cell_structure: break

            if valid_cell_structure:
                yield_matrix[i][j] = current_yield
                if num_cf + num_tft < 20:
                    logger.debug(f"Yield for CF {cf_panel.get('id', i)} - TFT {tft_panel.get('id', j)}: {current_yield}")
            # else: yield_matrix[i][j]는 이미 -1로 설정됨

    # --- 2. OR-Tools MIP 모델 구성 ---
    solver_name = 'CBC'  # 기본 솔버
    try:
        # SCIP이 더 성능이 좋을 수 있으나, 설치가 필요할 수 있음
        # solver = pywraplp.Solver.CreateSolver('SCIP')
        # if not solver:
        # logger.info("SCIP solver not found, attempting to use CBC.")
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            logger.error(f"{solver_name} 솔버를 생성할 수 없습니다. OR-Tools 설치를 확인하세요.")
            return [], 0, f"오류: MIP 솔버({solver_name})를 생성할 수 없습니다."
    except Exception as e:
        logger.error(f"솔버 생성 중 예외 발생: {e}", exc_info=True)
        return [], 0, f"오류: 솔버 생성 중 예외 발생 - {str(e)}"

    logger.info(f"Using {solver.SolverVersion()} for matching.")

    # --- 3. 변수 생성 (X_ij) ---
    x = {}
    for i in range(num_cf):
        for j in range(num_tft):
            if yield_matrix[i][j] >= 0:  # 유효한 매칭 쌍에 대해서만 변수 생성
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    if not x:  # 매칭 가능한 유효한 쌍이 하나도 없는 경우
        logger.warning("No valid pairs found to create decision variables.")
        return [], 0, "매칭 가능한 유효한 패널 쌍이 없습니다."

    # --- 4. 제약 조건 설정 ---
    for i in range(num_cf):
        solver.Add(sum(x[i, j] for j in range(num_tft) if (i, j) in x) <= 1)

    for j in range(num_tft):
        solver.Add(sum(x[i, j] for i in range(num_cf) if (i, j) in x) <= 1)

    # --- 5. 목표 함수 설정 ---
    objective = solver.Objective()
    for i in range(num_cf):
        for j in range(num_tft):
            if (i, j) in x:
                objective.SetCoefficient(x[i, j], float(yield_matrix[i][j]))  # Ensure profit is float
    objective.SetMaximization()

    # --- 6. 문제 해결 ---
    logger.info("Solving the MIP model...")
    status = solver.Solve()
    logger.info(f"Solver status: {status}")

    # --- 7. 결과 추출 ---
    matched_pairs_info = []
    total_yield_val = 0
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            msg = "Feasible solution found, but it might not be optimal."
            logger.warning(msg)
            error_msg = msg

        raw_objective_value = solver.Objective().Value()
        total_yield_val = raw_objective_value if raw_objective_value is not None else 0.0
        logger.info(f"Objective value (Total Yield): {total_yield_val}")

        for i in range(num_cf):
            for j in range(num_tft):
                if (i, j) in x and x[i, j].solution_value() > 0.5:
                    matched_pairs_info.append({
                        'cf': cf_panels[i],
                        'tft': tft_panels[j],
                        'cf_id': cf_panels[i].get('id', f'CF{i + 1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j + 1}'),
                        'yield_value': yield_matrix[i][j]
                    })
    else:
        solver_status_map = {
            pywraplp.Solver.OPTIMAL: "Optimal solution found.",  # 이미 위에서 처리됨
            pywraplp.Solver.FEASIBLE: "Feasible solution found.",  # 이미 위에서 처리됨
            pywraplp.Solver.INFEASIBLE: "문제가 실행 불가능(Infeasible)합니다. 데이터 또는 제약 조건을 확인하세요.",
            pywraplp.Solver.UNBOUNDED: "문제가 무한(Unbounded)합니다. 목표 함수나 제약 조건에 오류가 있을 수 있습니다.",
            pywraplp.Solver.ABNORMAL: "솔버가 비정상적으로 종료되었습니다. 입력 데이터나 모델에 문제가 있을 수 있습니다.",
            pywraplp.Solver.MODEL_INVALID: "모델이 유효하지 않습니다. 변수나 제약 조건 설정을 확인하세요.",
            pywraplp.Solver.NOT_SOLVED: "솔버가 문제를 풀지 못했습니다. 시간 제한 또는 다른 내부 문제일 수 있습니다."
        }
        error_msg = solver_status_map.get(status, f"매칭 해를 찾지 못했습니다. (솔버 상태 코드: {status})")
        logger.error(f"Solver failed. Status: {status}. Message: {error_msg}")
    solver_time_ms = solver.wall_time()/1000  # 밀리초
    return matched_pairs_info, total_yield_val, error_msg, solver_time_ms


def run_matching_transport_optimizer(input_data):
    """
    OR-Tools의 LinearSumAssignment 솔버를 사용하여 작업 배정 문제를 해결합니다.
    cost_matrix: 비용 행렬 (리스트의 리스트)
    """
    logger.info("Running Assignment Problem Optimizer.")
    logger.debug(f"Cost Matrix: {input_data}")

    workers = input_data['driver_names']
    if len(workers) == 0:
        return [], 0, "Error: There is no workers."

    tasks = input_data['zone_names']
    if len(tasks) == 0:
        return [], 0, "Error: There is no tasks."

    costs=input_data['cost_matrix']
    num_workers = len(costs)
    num_tasks = len(costs[0])
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x ={}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i,j] = solver.IntVar(0, 1, f'{workers[i]}_{tasks[j]}')

    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) == 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    logger.info("Solving the assignment model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time_ms:.2f} ms")

    results = {'assignments':[], 'total_cost':0}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or pywraplp.Solver.FEASIBLE:
        results['total_cost'] = solver.Objective().Value()
        logger.info(f"Total cost = {results['total_cost']}")
        for i in range(num_workers):
            for j in range(num_tasks):
                cost = costs[i][j]
                if x[i, j].solution_value() > 0.5:
                    results['assignments'].append({
                        'worker_name': workers[i],
                        'task_name': tasks[j],
                        'cost':cost
                    })
                    logger.debug(f'Worker {i} assigned to task {j} with a cost of {cost}')

    elif status == pywraplp.Solver.INFEASIBLE:
        error_msg = "실행 불가능한 문제입니다. 모든 작업자/작업 쌍에 대한 비용이 정의되었는지 확인하세요."
    else:
        error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"

    if error_msg:
        logger.error(f"Assignment optimization failed: {error_msg}")

    return results, error_msg, processing_time_ms
