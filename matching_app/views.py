from django.shortcuts import render
import json
import random
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver


# --- run_matching_algorithm 함수 정의 ---
def run_matching_algorithm(cf_panels, tft_panels):
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)

    if num_cf == 0 or num_tft == 0:
        return [], 0, "오류: CF 또는 TFT 패널 데이터가 없습니다."

    # --- 1. 수율 매트릭스 (C_ij) 계산 ---
    yield_matrix = [[0] * num_tft for _ in range(num_cf)]

    for i in range(num_cf):
        cf_panel = cf_panels[i]
        cf_map = cf_panel.get('defect_map', [])
        cf_rows = cf_panel.get('rows', 0)
        cf_cols = cf_panel.get('cols', 0)

        if not cf_map or cf_rows == 0 or cf_cols == 0:
            for j in range(num_tft):
                yield_matrix[i][j] = -1
            continue

        for j in range(num_tft):
            tft_panel = tft_panels[j]
            tft_map = tft_panel.get('defect_map', [])
            tft_rows = tft_panel.get('rows', 0)
            tft_cols = tft_panel.get('cols', 0)

            if not tft_map or tft_rows == 0 or tft_cols == 0:
                yield_matrix[i][j] = -1
                continue

            if cf_rows != tft_rows or cf_cols != tft_cols:
                yield_matrix[i][j] = -1
                continue

            current_yield = 0
            valid_pair = True
            for r in range(cf_rows):
                if not valid_pair: break
                for c in range(cf_cols):
                    if not (r < len(cf_map) and c < len(cf_map[r]) and \
                            r < len(tft_map) and c < len(tft_map[r])):
                        yield_matrix[i][j] = -1
                        valid_pair = False
                        break

                    is_cf_cell_good = (cf_map[r][c] == 0)
                    is_tft_cell_good = (tft_map[r][c] == 0)
                    if is_cf_cell_good and is_tft_cell_good:
                        current_yield += 1

            if valid_pair:
                yield_matrix[i][j] = current_yield
            # else: yield_matrix[i][j] is already -1 due to invalid structure

    # --- 2. OR-Tools MIP 모델 구성 ---
    try:
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')  # Fallback to CBC
            if not solver:
                return [], 0, "오류: MIP 솔버(SCIP 또는 CBC)를 생성할 수 없습니다. OR-Tools 설치를 확인하세요."
    except Exception as e:
        # This might catch issues if OR-Tools itself is not properly installed/configured
        return [], 0, f"오류: 솔버 생성 중 예외 발생 - {str(e)}"

    # --- 3. 변수 생성 (X_ij) ---
    x = {}
    for i in range(num_cf):
        for j in range(num_tft):
            if yield_matrix[i][j] >= 0:  # 유효한 매칭 쌍에 대해서만 변수 생성
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')

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
    status = solver.Solve()

    # --- 7. 결과 추출 ---
    matched_pairs_info = []
    total_yield_val = 0
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            error_msg = "최적해를 찾았지만, 더 좋은 해가 있을 수 있습니다 (Feasible solution)."

        # solver.Objective().Value() can sometimes be a very small negative number for 0 if precision issues
        # or if all yields are 0.
        raw_objective_value = solver.Objective().Value()
        total_yield_val = round(raw_objective_value) if raw_objective_value is not None else 0

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
            pywraplp.Solver.INFEASIBLE: "문제가 실행 불가능(Infeasible)합니다. 제약 조건을 확인하세요.",
            pywraplp.Solver.UNBOUNDED: "문제가 무한(Unbounded)합니다.",
            pywraplp.Solver.NOT_SOLVED: "솔버가 문제를 풀지 못했습니다 (Not Solved).",
            pywraplp.Solver.ABNORMAL: "솔버가 비정상적으로 종료되었습니다 (Abnormal).",
            pywraplp.Solver.MODEL_INVALID: "모델이 유효하지 않습니다 (Model Invalid)."
            # Add other statuses as needed
        }
        error_msg = solver_status_map.get(status, f"매칭 해를 찾지 못했습니다. (Solver status: {status})")

    return matched_pairs_info, total_yield_val, error_msg


# --- run_matching_algorithm 함수 정의 끝 ---


def matching_data_generation_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'data_generation',
        'cf_tft_panel_range': range(3, 11),
        'cell_dimension_range': range(3, 6),
    }
    # GET 요청시 form_values가 없으므로, 템플릿에서 None 체크를 하거나 빈 dict 전달
    context['form_values'] = {}

    if request.method == 'POST':
        context['form_values'] = request.POST  # POST시에는 실제 폼 값 전달
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

        except ValueError:
            context['error_message'] = "잘못된 입력입니다. 모든 숫자가 올바르게 입력되었는지 확인하세요."
        except Exception as e:
            context['error_message'] = f"데이터 생성 중 오류 발생: {str(e)}"

    return render(request, 'matching_app/data_generation.html', context)


def matching_small_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'small_scale_demo'
    }
    if request.method == 'POST':
        test_data_json_str = request.POST.get('test_data_json')
        context['submitted_json_data'] = test_data_json_str

        if test_data_json_str:
            try:
                test_data = json.loads(test_data_json_str)
                cf_panels = test_data.get('cf_panels')
                tft_panels = test_data.get('tft_panels')

                if cf_panels is None or tft_panels is None:
                    context['error_message'] = "오류: JSON 데이터에 'cf_panels' 또는 'tft_panels' 키가 없습니다."
                elif not isinstance(cf_panels, list) or not isinstance(tft_panels, list):
                    context['error_message'] = "오류: 'cf_panels' 와 'tft_panels'는 리스트여야 합니다."
                else:
                    valid_data = True
                    for p_list_name, p_list_items in [('CF', cf_panels), ('TFT', tft_panels)]:
                        if not valid_data: break
                        for p_idx, p_item in enumerate(p_list_items):
                            if not isinstance(p_item, dict) or \
                                    not all(k in p_item for k in ('id', 'rows', 'cols', 'defect_map')):
                                context[
                                    'error_message'] = f"오류: {p_list_name} 패널 데이터 (인덱스 {p_idx}, ID: {p_item.get('id', 'N/A')})에 필수 키(id, rows, cols, defect_map)가 누락되었거나 형식이 잘못되었습니다."
                                valid_data = False
                                break
                            if not isinstance(p_item['defect_map'], list) or \
                                    not all(isinstance(row, list) for row in p_item['defect_map']):  # 각 행도 리스트인지 확인
                                context[
                                    'error_message'] = f"오류: {p_list_name} 패널 {p_item.get('id')}의 defect_map 형식이 잘못되었습니다 (2차원 리스트여야 함)."
                                valid_data = False
                                break
                            # defect_map 내부 요소 타입 검사 (0 또는 1) - 선택 사항이지만 추가하면 좋음
                            for r_idx, row_data in enumerate(p_item['defect_map']):
                                if p_item['rows'] != len(p_item['defect_map']) or (
                                        len(row_data) > 0 and p_item['cols'] != len(row_data)):  # 행/열 개수 일치 확인
                                    context[
                                        'error_message'] = f"오류: {p_list_name} 패널 {p_item.get('id')}의 defect_map 크기가 명시된 rows/cols와 일치하지 않습니다."
                                    valid_data = False
                                    break
                                if not all(cell_val in (0, 1) for cell_val in row_data):
                                    context[
                                        'error_message'] = f"오류: {p_list_name} 패널 {p_item.get('id')}의 defect_map에 유효하지 않은 값(0 또는 1이 아님)이 포함되어 있습니다 (행 {r_idx})."
                                    valid_data = False
                                    break
                            if not valid_data: break

                    if valid_data:
                        context['input_cf_panels'] = cf_panels
                        context['input_tft_panels'] = tft_panels

                        matched_pairs, total_yield, error_msg = run_matching_algorithm(cf_panels, tft_panels)

                        if error_msg:
                            context['error_message'] = error_msg
                        else:
                            context['matching_pairs'] = matched_pairs
                            context['total_yield'] = total_yield
                            if matched_pairs or total_yield > 0:  # 실제로 매칭이 되었거나 수율이 있을 때만 성공 메시지
                                context['success_message'] = f"매칭 완료! 총 수율: {total_yield:.0f}"
                            elif not error_msg:  # 에러는 없는데 매칭 결과가 없는 경우
                                context['success_message'] = "매칭 가능한 쌍이 없거나 모든 쌍의 수율이 0입니다."

            except json.JSONDecodeError:
                context['error_message'] = "오류: 잘못된 JSON 형식입니다."
            except ValueError as ve:
                context['error_message'] = f"데이터 유효성 검사 또는 처리 오류: {str(ve)}"
            except Exception as e:
                context['error_message'] = f"매칭 중 예상치 못한 오류 발생: {str(e)}"
        else:
            context['error_message'] = "오류: 테스트 데이터가 제공되지 않았습니다."

    return render(request, 'matching_app/small_scale_demo.html', context)


def matching_large_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'large_scale_demo'
    }
    if request.method == 'POST':
        context['error_message'] = "대규모 매칭 로직은 아직 구현되지 않았습니다."
    return render(request, 'matching_app/large_scale_demo.html', context)