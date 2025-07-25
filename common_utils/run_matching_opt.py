from common_utils.common_run_opt import *
import logging

logger = logging.getLogger('matching_assignment_app')


def run_matching_cf_tft_algorithm(cf_panels, tft_panels):
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)

    logger.info(f"Starting matching algorithm for {num_cf} CF panel(s) and {num_tft} TFT panel(s).")

    if num_cf == 0 or num_tft == 0:
        msg = "Matching algorithm called with zero CF or TFT panels."
        logger.warning(msg)
        return [], 0, msg

    # --- 1. 수율 매트릭스 (C_ij) 계산 ---
    yield_matrix = create_cost_matrix(cf_panels, tft_panels)

    # --- 2. OR-Tools MIP 모델 구성 ---
    solver_name = 'CBC'  # 기본 솔버
    try:
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
        results = {'matched_pairs': [], 'total_yield_val': 0}

        raw_objective_value = solver.Objective().Value()
        results['total_yield'] = raw_objective_value if raw_objective_value is not None else 0.0

        for i in range(num_cf):
            for j in range(num_tft):
                if (i, j) in x and x[i, j].solution_value() > 0.5:
                    results['matched_pairs'].append({
                        'cf': cf_panels[i],
                        'tft': tft_panels[j],
                        'cf_id': cf_panels[i].get('id', f'CF{i + 1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j + 1}'),
                        'yield_value': yield_matrix[i][j]
                    })
    else:
        error_msg = f"매칭 해를 찾지 못했습니다. (솔버 상태 코드: {status})"
        logger.error(f"Solver failed. Status: {status}. Message: {error_msg}")
    solver_time_ms = solver.wall_time()/1000  # 밀리초
    return results, error_msg, solver_time_ms

def create_cost_matrix(cf_panels, tft_panels):
    logger.debug("Calculating yield matrix...")
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)
    yield_matrix = [[-1] * num_tft for _ in range(num_cf)]
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
                    logger.debug(
                        f"Yield for CF {cf_panel.get('id', i)} - TFT {tft_panel.get('id', j)}: {current_yield}")
    return yield_matrix