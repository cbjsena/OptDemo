# OptDemo/matching_app/views.py

from django.shortcuts import render
import json
import random
from ortools.linear_solver import pywraplp # OR-Tools MIP solver

def run_matching_algorithm(cf_panels, tft_panels):
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)

    if num_cf == 0 or num_tft == 0:
        return [], 0, "오류: CF 또는 TFT 패널 데이터가 없습니다."

    # --- 1. 수율 매트릭스 (C_ij) 계산 ---
    # C_ij = CF 패널 i와 TFT 패널 j를 매칭했을 때의 양품 셀 개수
    # 결함맵: 0 = 양품, 1 = 결함
    yield_matrix = [[0] * num_tft for _ in range(num_cf)]

    for i in range(num_cf):
        cf_panel = cf_panels[i]
        cf_map = cf_panel.get('defect_map', [])
        # 각 패널의 row, col 정보가 다를 수 있으므로 패널별로 가져옴
        cf_rows = cf_panel.get('rows', 0)
        cf_cols = cf_panel.get('cols', 0)

        if not cf_map or cf_rows == 0 or cf_cols == 0:
            # 해당 CF 패널 데이터가 유효하지 않으면 이 CF 패널에 대한 모든 매칭 수율은 0으로 처리
            # 또는 오류를 발생시킬 수 있습니다.
            for j in range(num_tft):
                yield_matrix[i][j] = -1 # 매칭 불가능 표시 (매우 낮은 값)
            continue


        for j in range(num_tft):
            tft_panel = tft_panels[j]
            tft_map = tft_panel.get('defect_map', [])
            tft_rows = tft_panel.get('rows', 0)
            tft_cols = tft_panel.get('cols', 0)

            if not tft_map or tft_rows == 0 or tft_cols == 0:
                yield_matrix[i][j] = -1
                continue

            # 두 패널의 크기가 다르면 매칭 불가능 (수율 0 또는 매우 낮은 값)
            if cf_rows != tft_rows or cf_cols != tft_cols:
                yield_matrix[i][j] = -1 # 매칭 불가능 (매우 낮은 값으로 설정하여 선택되지 않도록)
                continue
            
            current_yield = 0
            for r in range(cf_rows):
                for c in range(cf_cols):
                    # defect_map의 각 행이 cf_cols 길이를 가지고 있는지, r, c가 범위 내인지 확인
                    if r < len(cf_map) and c < len(cf_map[r]) and \
                       r < len(tft_map) and c < len(tft_map[r]):
                        is_cf_cell_good = (cf_map[r][c] == 0)
                        is_tft_cell_good = (tft_map[r][c] == 0)
                        if is_cf_cell_good and is_tft_cell_good:
                            current_yield += 1
                    else:
                        # 결함맵 데이터 구조 오류 처리
                        yield_matrix[i][j] = -1 # 이 쌍은 매칭 불가능으로 처리
                        current_yield = -1 # 루프 탈출 또는 플래그 설정용
                        break 
                if current_yield == -1:
                    break
            
            if current_yield != -1:
                 yield_matrix[i][j] = current_yield

    # --- 2. OR-Tools MIP 모델 구성 ---
    # SCIP, CBC, GLPK, Gurobi 등 다양한 솔버 사용 가능 (설치 필요)
    # Create the mip solver with the SCIP backend.
    try:
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # SCIP이 없으면 CBC 시도
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                 return [], 0, "오류: MIP 솔버(SCIP 또는 CBC)를 생성할 수 없습니다. OR-Tools 설치를 확인하세요."
    except Exception as e:
        return [], 0, f"오류: 솔버 생성 중 예외 발생 - {str(e)}"


    # --- 3. 변수 생성 (X_ij) ---
    # x[i][j]는 CF_i와 TFT_j가 매칭되면 1, 아니면 0
    x = {}
    for i in range(num_cf):
        for j in range(num_tft):
            if yield_matrix[i][j] >= 0: # 유효한 매칭 쌍에 대해서만 변수 생성
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # --- 4. 제약 조건 설정 ---
    # 각 CF 패널은 최대 하나의 TFT 패널과 매칭
    for i in range(num_cf):
        solver.Add(sum(x[i, j] for j in range(num_tft) if (i,j) in x) <= 1)

    # 각 TFT 패널은 최대 하나의 CF 패널과 매칭
    for j in range(num_tft):
        solver.Add(sum(x[i, j] for i in range(num_cf) if (i,j) in x) <= 1)

    # --- 5. 목표 함수 설정 ---
    objective = solver.Objective()
    for i in range(num_cf):
        for j in range(num_tft):
            if (i,j) in x: # 유효한 변수에 대해서만 계수 설정
                objective.SetCoefficient(x[i, j], yield_matrix[i][j])
    objective.SetMaximization()

    # --- 6. 문제 해결 ---
    status = solver.Solve()

    # --- 7. 결과 추출 ---
    matched_pairs_info = []
    total_yield_val = 0
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL:
        total_yield_val = solver.Objective().Value()
        for i in range(num_cf):
            for j in range(num_tft):
                if (i,j) in x and x[i, j].solution_value() > 0.5: # X_ij가 1인 경우
                    matched_pairs_info.append({
                        'cf': cf_panels[i], 
                        'tft': tft_panels[j],
                        'cf_id': cf_panels[i].get('id', f'CF{i+1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j+1}'),
                        'yield_value': yield_matrix[i][j]
                    })
    elif status == pywraplp.Solver.FEASIBLE:
        total_yield_val = solver.Objective().Value() # 최적은 아니지만 가능한 해
        # (위와 동일하게 결과 추출)
        for i in range(num_cf):
            for j in range(num_tft):
                if (i,j) in x and x[i, j].solution_value() > 0.5:
                     matched_pairs_info.append({
                        'cf_id': cf_panels[i].get('id', f'CF{i+1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j+1}'),
                        'yield_value': yield_matrix[i][j]
                    })
        error_msg = "최적해를 찾았지만, 더 좋은 해가 있을 수 있습니다 (Feasible solution)."
    else:
        error_msg = "매칭 해를 찾지 못했습니다. (Solver status: " + str(status) + ")"
        if status == pywraplp.Solver.INFEASIBLE:
            error_msg = "문제가 실행 불가능(Infeasible)합니다. 제약 조건을 확인하세요."
        elif status == pywraplp.Solver.UNBOUNDED:
             error_msg = "문제가 무한(Unbounded)합니다."
        elif status == pywraplp.Solver.NOT_SOLVED:
             error_msg = "솔버가 문제를 풀지 못했습니다."


    return matched_pairs_info, total_yield_val, error_msg


def matching_data_generation_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'data_generation',
        'cf_tft_panel_range': range(3, 11),
        'cell_dimension_range': range(3, 6),
    }
    if request.method == 'POST':
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
                context['form_values'] = request.POST 
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
            context['form_values'] = request.POST 
        except ValueError:
            context['error_message'] = "잘못된 입력입니다. 모든 숫자가 올바르게 입력되었는지 확인하세요."
            context['form_values'] = request.POST 
        except Exception as e:
            context['error_message'] = f"데이터 생성 중 오류 발생: {str(e)}"
            context['form_values'] = request.POST 
    return render(request, 'matching_app/data_generation.html', context)


def matching_small_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'small_scale_demo'
    }
    if request.method == 'POST':
        test_data_json_str = request.POST.get('test_data_json')
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
                    # 모든 패널에 id, rows, cols, defect_map이 있는지 간단히 확인 (더욱 강력한 유효성 검사 필요 가능)
                    for p_list in [cf_panels, tft_panels]:
                        for p_idx, p_item in enumerate(p_list):
                            if not all(k in p_item for k in ('id', 'rows', 'cols', 'defect_map')):
                                raise ValueError(f"패널 데이터 {p_idx}에 필수 키가 누락되었습니다.")
                            if not isinstance(p_item['defect_map'], list):
                                raise ValueError(f"패널 {p_item.get('id')}의 defect_map이 리스트가 아닙니다.")


                    # OR-Tools를 사용하여 매칭 알고리즘 실행
                    matched_pairs, total_yield, error_msg = run_matching_algorithm(cf_panels, tft_panels)
                    
                    if error_msg:
                        context['error_message'] = error_msg
                    else:
                        context['matching_pairs'] = matched_pairs
                        context['total_yield'] = total_yield
                        # 성공 메시지 또는 추가 정보
                        context['success_message'] = f"매칭 완료! 총 수율: {total_yield}"
                
                # 폼에 입력된 JSON 데이터 유지
                context['submitted_json_data'] = test_data_json_str

            except json.JSONDecodeError:
                context['error_message'] = "오류: 잘못된 JSON 형식입니다."
                context['submitted_json_data'] = test_data_json_str
            except ValueError as ve: # 데이터 구조 관련 오류 처리
                context['error_message'] = f"데이터 유효성 검사 오류: {str(ve)}"
                context['submitted_json_data'] = test_data_json_str
            except Exception as e:
                context['error_message'] = f"매칭 중 오류 발생: {str(e)}"
                context['submitted_json_data'] = test_data_json_str
        else:
            context['error_message'] = "오류: 테스트 데이터가 제공되지 않았습니다."

    return render(request, 'matching_app/small_scale_demo.html', context)


def matching_large_scale_demo_view(request):
    context = {
        'active_model': 'Matching',
        'active_submenu': 'large_scale_demo'
    }
    if request.method == 'POST':
        # 대량 데이터 처리 로직 (파일 업로드 등)
        # run_matching_algorithm 함수 재활용 가능
        context['error_message'] = "대규모 매칭 로직은 아직 구현되지 않았습니다."
    return render(request, 'matching_app/large_scale_demo.html', context)