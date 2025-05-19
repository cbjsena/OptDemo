from django.shortcuts import render
import json
import random
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver
import logging
logger = logging.getLogger(__name__)

# --- run_matching_algorithm 함수 정의 ---
def run_matching_algorithm(cf_panels, tft_panels):
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

    return matched_pairs_info, total_yield_val, error_msg


def matching_data_generation_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'data_generation',
        'cf_tft_panel_range': range(3, 11),
        'cell_dimension_range': range(3, 6),
        'form_values': request.POST if request.method == 'POST' else {},  # GET 요청 시 빈 dict
    }

    if request.method == 'POST':
        logger.debug(f"Data generation POST request received. Data: {request.POST}")
        try:
            num_cf_panels = int(request.POST.get('num_cf_panels', 5))
            num_tft_panels = int(request.POST.get('num_tft_panels', 5))
            panel_rows = int(request.POST.get('panel_rows', 3))
            panel_cols = int(request.POST.get('panel_cols', 3))
            defect_rate = int(request.POST.get('defect_rate', 10))

            if not (3 <= num_cf_panels <= 10 and \
                    3 <= num_tft_panels <= 10 and \
                    3 <= panel_rows <= 5 and \
                    3 <= panel_cols <= 5 and \
                    0 <= defect_rate <= 100):
                context['error_message'] = "입력값이 허용 범위를 벗어났습니다."
                logger.warning(f"Invalid input for data generation: {request.POST}")
                return render(request, 'matching_app/data_generation.html', context)

            def create_panel_data(panel_id_prefix, num_panels, rows, cols, rate):
                panels = []
                for i_panel in range(1, num_panels + 1):
                    defect_map = []
                    for r_idx in range(rows):
                        row_map = []
                        for c_idx in range(cols):
                            if random.randint(1, 100) <= rate:
                                row_map.append(1)
                            else:
                                row_map.append(0)
                        defect_map.append(row_map)
                    panels.append({
                        "id": f"{panel_id_prefix}{i_panel}",
                        "rows": rows,
                        "cols": cols,
                        "defect_map": defect_map
                    })
                return panels

            generated_cf_panels = create_panel_data("CF", num_cf_panels, panel_rows, panel_cols, defect_rate)
            generated_tft_panels = create_panel_data("TFT", num_tft_panels, panel_rows, panel_cols, defect_rate)

            generated_data = {
                "panel_dimensions": {"rows": panel_rows, "cols": panel_cols},
                "cf_panels": generated_cf_panels,
                "tft_panels": generated_tft_panels,
                "settings": {
                    "num_cf_panels": num_cf_panels,
                    "num_tft_panels": num_tft_panels,
                    "defect_rate_percent": defect_rate,
                    "panel_rows": panel_rows,
                    "panel_cols": panel_cols,
                }
            }
            context['generated_data'] = generated_data
            context['generated_data_json_pretty'] = json.dumps(generated_data, indent=4)
            logger.info("Panel data generated successfully.")
        except ValueError as e:
            context['error_message'] = "잘못된 입력입니다. 모든 숫자가 올바르게 입력되었는지 확인하세요."
            logger.error(f"ValueError during data generation: {e}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"데이터 생성 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error during data generation: {e}", exc_info=True)

    return render(request, 'matching_app/data_generation.html', context)


def matching_small_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'small_scale_demo'
    }
    if request.method == 'POST':
        test_data_json_str = request.POST.get('test_data_json')
        context['submitted_json_data'] = test_data_json_str
        logger.info("Small scale demo POST request received.")
        logger.debug(f"Submitted JSON data: {test_data_json_str[:200]}...")  # 너무 길면 일부만 로깅

        if test_data_json_str:
            try:
                test_data = json.loads(test_data_json_str)
                cf_panels = test_data.get('cf_panels')
                tft_panels = test_data.get('tft_panels')

                # --- 강화된 데이터 유효성 검사 ---
                if cf_panels is None or tft_panels is None:
                    msg = "오류: JSON 데이터에 'cf_panels' 또는 'tft_panels' 키가 없습니다."
                    logger.error(msg)
                    context['error_message'] = msg
                    return render(request, 'matching_app/small_scale_demo.html', context)

                if not isinstance(cf_panels, list) or not isinstance(tft_panels, list):
                    msg = "오류: 'cf_panels' 와 'tft_panels'는 리스트여야 합니다."
                    logger.error(msg)
                    context['error_message'] = msg
                    return render(request, 'matching_app/small_scale_demo.html', context)

                # 패널 데이터 구조 유효성 검사 함수
                def validate_panel_list(panel_list_items, panel_type_name):
                    for p_idx, p_item in enumerate(panel_list_items):
                        if not isinstance(p_item, dict) or \
                                not all(k in p_item for k in ('id', 'rows', 'cols', 'defect_map')):
                            return f"오류: {panel_type_name} 패널 데이터 (인덱스 {p_idx}, ID: {p_item.get('id', 'N/A')})에 필수 키(id, rows, cols, defect_map)가 누락되었거나 형식이 잘못되었습니다."
                        if not (isinstance(p_item['rows'], int) and p_item['rows'] > 0 and isinstance(p_item['cols'],
                                                                                                      int) and p_item[
                                    'cols'] > 0):
                            return f"오류: {panel_type_name} 패널 {p_item.get('id')}의 rows/cols 값이 유효한 양의 정수가 아닙니다."
                        if not isinstance(p_item['defect_map'], list) or len(p_item['defect_map']) != p_item['rows']:
                            return f"오류: {panel_type_name} 패널 {p_item.get('id')}의 defect_map 형식이 잘못되었거나 행 수가 명시된 rows와 일치하지 않습니다."
                        for r_idx, row_data in enumerate(p_item['defect_map']):
                            if not isinstance(row_data, list) or len(row_data) != p_item['cols']:
                                return f"오류: {panel_type_name} 패널 {p_item.get('id')}의 defect_map의 행 {r_idx}의 열 수가 명시된 cols와 일치하지 않습니다."
                            if not all(cell_val in (0, 1) for cell_val in row_data):
                                return f"오류: {panel_type_name} 패널 {p_item.get('id')}의 defect_map에 유효하지 않은 값(0 또는 1이 아님)이 포함되어 있습니다 (행 {r_idx})."
                    return None  # 유효성 검사 통과

                validation_error_cf = validate_panel_list(cf_panels, "CF")
                if validation_error_cf:
                    logger.error(f"CF Panel Validation Error: {validation_error_cf}")
                    context['error_message'] = validation_error_cf
                    return render(request, 'matching_app/small_scale_demo.html', context)

                validation_error_tft = validate_panel_list(tft_panels, "TFT")
                if validation_error_tft:
                    logger.error(f"TFT Panel Validation Error: {validation_error_tft}")
                    context['error_message'] = validation_error_tft
                    return render(request, 'matching_app/small_scale_demo.html', context)

                # 유효성 검사 통과 후
                context['input_cf_panels'] = cf_panels
                context['input_tft_panels'] = tft_panels
                logger.info("Input panel data validated successfully.")

                matched_pairs, total_yield, error_msg = run_matching_algorithm(cf_panels, tft_panels)

                if error_msg:
                    context['error_message'] = error_msg
                    logger.error(f"Error from matching algorithm: {error_msg}")
                else:
                    context['matching_pairs'] = matched_pairs
                    context['total_yield'] = total_yield
                    if matched_pairs or total_yield > 0:
                        msg = f"매칭 완료! 총 수율: {total_yield:.0f}"
                        context['success_message'] = msg
                        logger.info(msg)
                    elif not error_msg:  # 에러 없고 매칭 결과도 없을 때
                        msg = "매칭 가능한 쌍이 없거나 모든 쌍의 수율이 0입니다."
                        context['success_message'] = msg  # 정보성 메시지로 처리
                        logger.info(msg)

            except json.JSONDecodeError as e:
                msg = "오류: 잘못된 JSON 형식입니다."
                logger.error(f"{msg} - {e}", exc_info=True)
                context['error_message'] = msg
            except ValueError as ve:  # 직접 발생시킨 ValueError 포함
                msg = f"데이터 유효성 검사 또는 처리 오류: {str(ve)}"
                logger.error(msg, exc_info=True)
                context['error_message'] = msg
            except Exception as e:
                msg = f"매칭 중 예상치 못한 오류 발생: {str(e)}"
                logger.error(msg, exc_info=True)
                context['error_message'] = msg
        else:
            context['error_message'] = "오류: 테스트 데이터가 제공되지 않았습니다."
            logger.warning("No test data provided for small scale demo.")

    return render(request, 'matching_app/small_scale_demo.html', context)


def matching_large_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'large_scale_demo'
    }
    if request.method == 'POST':
        context['error_message'] = "대규모 매칭 로직은 아직 구현되지 않았습니다."
    return render(request, 'matching_app/large_scale_demo.html', context)