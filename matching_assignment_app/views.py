from django.shortcuts import render

# Create your views here.
from django.conf import settings
from django.shortcuts import render

import json
import random
import os
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver

import logging

logger = logging.getLogger(__name__)

def validate_panel_data_structure(panel_list_items, panel_type_name):
    """
    패널 데이터 리스트의 구조와 내용의 유효성을 검사합니다.
    오류가 있으면 오류 메시지 문자열을, 정상이면 None을 반환합니다.
    """
    if panel_list_items is None:
        return f"오류: '{panel_type_name}_panels' 데이터가 없습니다 (None)."
    if not isinstance(panel_list_items, list):
        return f"오류: '{panel_type_name}_panels' 데이터가 리스트 형식이 아닙니다."
    if not panel_list_items:  # 빈 리스트도 유효할 수 있으나, 여기서는 패널이 있어야 한다고 가정
        return f"오류: '{panel_type_name}_panels' 리스트가 비어있습니다."

    for p_idx, p_item in enumerate(panel_list_items):
        if not isinstance(p_item, dict):
            return f"오류: {panel_type_name} 패널 데이터 (인덱스 {p_idx})가 딕셔너리 형식이 아닙니다."

        required_keys = ('id', 'rows', 'cols', 'defect_map')
        missing_keys = [k for k in required_keys if k not in p_item]
        if missing_keys:
            return f"오류: {panel_type_name} 패널 (ID: {p_item.get('id', 'N/A')}, 인덱스 {p_idx})에 필수 키가 누락되었습니다: {', '.join(missing_keys)}."

        panel_id = p_item.get('id', f'인덱스 {p_idx}')
        rows = p_item.get('rows')
        cols = p_item.get('cols')
        defect_map = p_item.get('defect_map')

        if not (isinstance(rows, int) and rows > 0 and isinstance(cols, int) and cols > 0):
            return f"오류: {panel_type_name} 패널 {panel_id}의 rows/cols 값이 유효한 양의 정수가 아닙니다 (rows: {rows}, cols: {cols})."

        if not isinstance(defect_map, list) or len(defect_map) != rows:
            return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map 형식이 잘못되었거나 행 수가 명시된 rows({rows})와 일치하지 않습니다 (실제 행 수: {len(defect_map) if isinstance(defect_map, list) else 'N/A'})."

        for r_idx, row_data in enumerate(defect_map):
            if not isinstance(row_data, list) or len(row_data) != cols:
                return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map의 행 {r_idx}의 열 수가 명시된 cols({cols})와 일치하지 않습니다 (실제 열 수: {len(row_data) if isinstance(row_data, list) else 'N/A'})."
            if not all(cell_val in (0, 1) for cell_val in row_data):
                return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map에 유효하지 않은 값(0 또는 1이 아님)이 포함되어 있습니다 (행 {r_idx})."
    return None  # 유효성 검사 통과

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


def create_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate):
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
    return generated_data


def lcd_cf_tft_introduction_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu': 'introduction',
    }
    return render(request, 'matching_assignment_app/lcd_cf_tft_introduction.html', context)

def lcd_cf_tft_data_generation_view(request):
    context = {
        'active_model': 'Matching & Assignment',
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

            generated_data = create_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate)
            context['generated_data'] = generated_data
            context['generated_data_json_pretty'] = json.dumps(generated_data, indent=4)
            logger.info("Panel data generated successfully.")
        except ValueError as e:
            context['error_message'] = "잘못된 입력입니다. 모든 숫자가 올바르게 입력되었는지 확인하세요."
            logger.error(f"ValueError during data generation: {e}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"데이터 생성 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error during data generation: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/lcd_cf_tft_data_generation.html', context)


def lcd_cf_tft_small_scale_demo_view(request):
    context = {
        'active_model': 'Matching & Assignment',
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

                # 공통 유효성 검사 함수 사용
                validation_error_cf = validate_panel_data_structure(cf_panels, "CF")
                if validation_error_cf:
                    logger.error(f"CF Panel Validation Error: {validation_error_cf}")
                    context['error_message'] = validation_error_cf
                    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)

                validation_error_tft = validate_panel_data_structure(tft_panels, "TFT")
                if validation_error_tft:
                    logger.error(f"TFT Panel Validation Error: {validation_error_tft}")
                    context['error_message'] = validation_error_tft
                    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)

                # 유효성 검사 통과 후
                context['input_cf_panels'] = cf_panels
                context['input_tft_panels'] = tft_panels
                logger.info("Input panel data validated successfully.")

                matched_pairs, total_yield, error_msg, solver_time = run_matching_algorithm(cf_panels, tft_panels)

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

    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)


def lcd_cf_tft_large_scale_demo_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu': 'large_scale_demo',
        'available_json_files': []
    }

    large_data_dir = getattr(settings, 'LARGE_SCALE_DATA_DIR', None)
    if large_data_dir and os.path.isdir(large_data_dir):
        try:
            files = [f for f in os.listdir(large_data_dir) if f.endswith('.json') and f.startswith('test_cf')]
            context['available_json_files'] = [{'value': f, 'name': f} for f in sorted(files, reverse=True)]
        except OSError as e:
            logger.error(f"Error listing files in {large_data_dir}: {e}")
            context['error_message'] = f"서버의 데이터 디렉토리에서 파일 목록을 읽어오는 데 실패했습니다."
    elif not large_data_dir:
        logger.warning("LARGE_SCALE_DATA_DIR is not defined in settings. File selection will not work.")
        # context['error_message'] = "서버 데이터 디렉토리가 설정되지 않았습니다." # 사용자에게 보여줄 필요는 없을 수도 있음

    if request.method == 'POST':
        logger.info(f"Large scale demo POST request received. Input type: {request.POST.get('large_data_input_type')}")
        input_type = request.POST.get('large_data_input_type')
        cf_panels = None
        tft_panels = None
        loaded_filename = None

        try:
            if input_type == 'make_json':
                logger.info("Processing 'make_json' input type.")
                num_cf = request.POST.get('num_cf_panels', '100')
                num_tft = request.POST.get('num_tft_panels', '100')
                panel_r = request.POST.get('panel_rows', '4')
                panel_c = request.POST.get('panel_cols', '4')
                defect_rate_str = request.POST.get('defect_rate', '10')

                # 입력값 유효성 검사 (숫자 변환 및 범위)
                num_cf = int(num_cf)
                num_tft = int(num_tft)
                panel_r = int(panel_r)
                panel_c = int(panel_c)
                defect_rate_percent  = int(defect_rate_str)

                generated_data = create_json_data(num_cf, num_tft, panel_r, panel_c, defect_rate_percent )
                if large_data_dir:
                    # 중복 방지를 위해 시퀀스 번호 또는 타임스탬프 사용
                    seq = 0
                    cf_panels = generated_data.get('cf_panels')
                    tft_panels = generated_data.get('tft_panels')

                    while True:
                        filename_pattern = f"test_cf{num_cf}_tft{num_tft}_row{panel_r}_col{panel_c}_rate{str(defect_rate_percent ).replace('.', 'p')}"
                        if seq == 0:
                            potential_filename = f"{filename_pattern}.json"
                        else:
                            potential_filename = f"{filename_pattern}_seq{seq}.json"

                        loaded_filename = potential_filename

                        filepath = os.path.join(large_data_dir, potential_filename)
                        if not os.path.exists(filepath):
                            loaded_filename = potential_filename  # 저장될 (또는 사용될) 파일명
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(generated_data, f, indent=2)
                            logger.info(f"Generated data saved to: {filepath}")
                            context['success_message'] = f"데이터가 생성되어 '{loaded_filename}'으로 서버에 저장되었습니다. 이제 매칭을 실행합니다."
                            # 파일 목록을 즉시 업데이트하기 위해 다시 로드 (선택 사항)
                            files = [f for f in os.listdir(large_data_dir) if
                                     f.endswith('.json') and f.startswith('test_cf')]
                            context['available_json_files'] = [{'value': f, 'name': f} for f in
                                                               sorted(files, reverse=True)]
                            break
                        seq += 1
                        if seq > 100:  # 무한 루프 방지
                            logger.error("Could not find a unique filename after 100 attempts for make_json.")
                            context['error_message'] = "생성된 데이터를 저장할 고유한 파일 이름을 찾는 데 실패했습니다."
                            return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)
                else:
                    logger.warning("LARGE_SCALE_DATA_DIR not set. Generated data will not be saved.")
                    context['info_message'] = "데이터가 생성되었지만, 서버 저장 경로가 설정되지 않아 저장되지 않았습니다. 매칭은 진행됩니다."


            elif input_type == 'select_json':
                logger.info("Processing 'select_json' input type.")
                selected_file = request.POST.get('selected_json_file')
                if not selected_file:
                    context['error_message'] = "서버에서 JSON 파일이 선택되지 않았습니다."
                elif not large_data_dir:
                    context['error_message'] = "서버 데이터 디렉토리가 설정되지 않아 파일을 로드할 수 없습니다."
                else:
                    filepath = os.path.join(large_data_dir, selected_file)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        cf_panels = data.get('cf_panels')
                        tft_panels = data.get('tft_panels')
                        loaded_filename = selected_file
                        logger.info(f"Data loaded from selected server file: {filepath}")
                    else:
                        context['error_message'] = f"선택한 파일 '{selected_file}'을 서버에서 찾을 수 없습니다."

            elif input_type == 'upload_json':
                logger.info("Processing 'upload_json' input type.")
                uploaded_file = request.FILES.get('data_file')
                if not uploaded_file:
                    context['error_message'] = "업로드된 JSON 파일이 없습니다."
                elif not uploaded_file.name.endswith('.json'):
                    context['error_message'] = "잘못된 파일 형식입니다. JSON 파일만 업로드 가능합니다."
                else:
                    try:
                        # FileSystemStorage를 사용하면 임시 파일 또는 메모리에서 바로 처리 가능
                        # fs = FileSystemStorage()
                        # filename = fs.save(uploaded_file.name, uploaded_file) # 임시 저장 (선택)
                        # filepath = fs.path(filename)
                        # with open(filepath, 'r', encoding='utf-8') as f:
                        #     data = json.load(f)
                        # fs.delete(filename) # 임시 파일 삭제

                        # 메모리에서 직접 읽기 (더 효율적)
                        data = json.load(uploaded_file)
                        cf_panels = data.get('cf_panels')
                        tft_panels = data.get('tft_panels')
                        loaded_filename = uploaded_file.name
                        logger.info(f"Data loaded from uploaded file: {uploaded_file.name}")
                    except json.JSONDecodeError:
                        context['error_message'] = "업로드된 JSON 파일의 형식이 올바르지 않습니다."
                    except Exception as e:
                        context['error_message'] = f"파일 처리 중 오류 발생: {str(e)}"
                        logger.error(
                            f"Error processing uploaded file {uploaded_file.name if uploaded_file else 'N/A'}: {e}",
                            exc_info=True)
            else:
                context['error_message'] = "알 수 없는 입력 유형입니다."

            # --- 데이터 로드 또는 생성 후 유효성 검사 및 매칭 실행 ---
            if cf_panels is not None and tft_panels is not None:
                validation_error_cf = validate_panel_data_structure(cf_panels, "CF")
                if validation_error_cf:
                    logger.error(f"Large Scale CF Panel Validation Error: {validation_error_cf}")
                    context['error_message'] = validation_error_cf
                    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)

                validation_error_tft = validate_panel_data_structure(tft_panels, "TFT")
                if validation_error_tft:
                    logger.error(f"Large Scale TFT Panel Validation Error: {validation_error_tft}")
                    context['error_message'] = validation_error_tft
                    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)

                logger.info(
                    f"Data for large scale matching validated. CF: {len(cf_panels)}, TFT: {len(tft_panels)}. Source: {loaded_filename}")

                matched_pairs, total_yield, error_msg, solver_time = run_matching_algorithm(cf_panels, tft_panels)

                if error_msg:
                    context['error_message'] = (context.get('error_message', '') + " " + error_msg).strip()
                else:
                    num_matches = len(matched_pairs)
                    avg_yield = total_yield / num_matches if num_matches > 0 else 0
                    processing_time_val=solver_time
                    context['large_scale_results'] = {
                        'num_cf': len(cf_panels),
                        'num_tft': len(tft_panels),
                        'num_matches': num_matches,
                        'total_yield': round(total_yield),
                        'avg_yield': avg_yield,
                        'processing_time_seconds': processing_time_val,
                        'sample_matches': matched_pairs[:10],  # 처음 10개만 샘플로
                        'source_file': loaded_filename if loaded_filename else "Newly Generated (unsaved or error saving)"
                    }
                    success_msg_main = f"대규모 매칭 완료 (소스: {loaded_filename})."
                    current_success = context.get('success_message_extra', '')  # 파일 저장 성공 메시지 등
                    context['success_message'] = (
                                current_success + " " + success_msg_main).strip() if current_success else success_msg_main
                    logger.info(
                        f"Large scale matching completed. Total yield: {total_yield}. Source: {loaded_filename}")


            elif not context.get('error_message'):  # 데이터 로드/생성 실패했고, 명시적 에러 메시지 없을 때
                context['error_message'] = "패널 데이터를 준비하지 못했습니다."
                logger.warning(
                    "Panel data could not be prepared for large scale matching and no specific error was set.")


        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in large_scale_demo_view: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)


def assignment_problem_introduction_view():
    return None


def stable_matching_introduction_view():
    return None


def resource_skill_matching_introduction_view():
    return None