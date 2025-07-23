from django.conf import settings
from django.shortcuts import render
import json

from common_utils.run_puzzle_opt import *
from common_utils.data_utils_puzzle import *
from core.decorators import log_view_activity

logger = logging.getLogger(__name__)


@log_view_activity
def main_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'main_introduction'
    }
    return render(request, 'puzzles_logic_app/puzzles_logic_introduction.html', context)


@log_view_activity
def diet_problem_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'diet_problem_introduction'
    }
    return render(request, 'puzzles_logic_app/diet_problem_introduction.html', context)


@log_view_activity
def diet_problem_demo_view(request):
    nutrients_list = []
    foods_list = []
    form_data = {}

    if request.method == 'GET':
        submitted_num_foods = int(request.GET.get('num_foods_to_show', preset_diet_food_number))
        submitted_num_foods = max(2, min(10, submitted_num_foods))
        submitted_num_nutrients = int(request.GET.get('num_nutrients_to_show', preset_diet_nutrient_number))
        submitted_num_nutrients = max(2, min(5, submitted_num_nutrients))

        for i in range(submitted_num_nutrients):
            preset = preset_diet_nutrients_data[i]
            nutrients_list.append({
                'name': request.GET.get(f'nutrient_{i}_name', preset['name']),
                'min': request.GET.get(f'nutrient_{i}_min', preset['min']),
                'max': request.GET.get(f'nutrient_{i}_max', preset['max']),
            })
        for i in range(submitted_num_foods):
            preset = preset_diet_foods_data[i]
            food_info = {
                'name': request.GET.get(f'food_{i}_name', preset['name']),
                'cost': request.GET.get(f'food_{i}_cost', preset['cost']),
                'min_intake': request.GET.get(f'food_{i}_min_intake', preset['min_intake']),
                'max_intake': request.GET.get(f'food_{i}_max_intake', preset['max_intake']),
                'nutrients': [request.GET.get(f'nutrient_val_{i}_{j}', preset['nutrients'][j]) for j in range(submitted_num_nutrients)]
            }
            foods_list.append(food_info)
    else: # POST
        form_data = request.POST.copy()
        submitted_num_foods = int(form_data.get('num_foods', preset_diet_food_number))
        submitted_num_foods = max(2, min(10, submitted_num_foods))
        submitted_num_nutrients = int(form_data.get('num_nutrients', preset_diet_nutrient_number))
        submitted_num_nutrients = max(2, min(5, submitted_num_nutrients))

        for i in range(submitted_num_nutrients):
            nutrients_list.append({'name': form_data.get(f'nutrient_{i}_name'), 'min': form_data.get(f'nutrient_{i}_min'), 'max': form_data.get(f'nutrient_{i}_max')})
        for i in range(submitted_num_foods):
            foods_list.append({
                'name': form_data.get(f'food_{i}_name'),
                'cost': form_data.get(f'food_{i}_cost'),
                'min_intake': form_data.get(f'food_{i}_min_intake'),
                'max_intake': form_data.get(f'food_{i}_max_intake'),
                'nutrients': [form_data.get(f'nutrient_val_{i}_{j}') for j in range(submitted_num_nutrients)]
            })

    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'Diet Problem Demo',
        'nutrients_list': nutrients_list,
        'foods_list': foods_list,
        'form_data': form_data,
        'results': None, 'manual_results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_foods_options': range(2, 11),
        'num_nutrients_options': range(2, 6),
        'submitted_num_foods': submitted_num_foods,
        'submitted_num_nutrients': submitted_num_nutrients,
    }

    if request.method == 'POST':
        action = form_data.get('action')
        try:
            if action == "optimize":    # 최적화 실행 버튼
                logger.info(f"Diet Problem Demo  {action} POST request processing.")
                # 1. 데이터 파일 새성 및 검증
                input_data = create_diet_json_data(form_data)

                # 2. 파일 저장
                if settings.SAVE_DATA_FILE:
                    success_save_message, save_error = save_puzzle_json_data(input_data)
                    if save_error:
                        context['error_message'] = save_error
                    elif success_save_message:
                        context['success_save_message'] = success_save_message

                # 3. 최적화 실행
                results_data, error_msg_opt, processing_time = run_diet_optimizer(input_data)
                context['processing_time_seconds'] = processing_time

                if error_msg_opt:
                    context['error_message'] = error_msg_opt
                elif results_data:
                    context['results'] = results_data
                    context['success_message'] = f"최적 식단 계산 완료! 최소 비용: {results_data['total_cost']}"
                    # 다음 수동 조회를 위해 원본 데이터와 결과를 숨겨진 필드로 전달할 수 있도록 저장
                    context['original_input_json'] = json.dumps(input_data)
                    context['optimal_results_json'] = json.dumps(results_data)
            elif action == "manual_check":
                logger.info(f"Flow Shop Demo {action} POST request processing.")

                # 숨겨진 필드에서 원본 데이터와 최적화 결과 로드
                original_input_str = form_data.get('original_input_json')
                optimal_results_str = form_data.get('optimal_results_json')
                if not original_input_str or not optimal_results_str:
                    raise ValueError("비교를 위한 원본 데이터 또는 최적화 결과가 없습니다.")

                original_input_data = json.loads(original_input_str)
                optimal_results_data = json.loads(optimal_results_str)
                context['results'] = optimal_results_data  # 최적 결과 다시 표시
                context['original_input_json'] = original_input_str
                context['optimal_results_json'] = optimal_results_str

                manual_intakes = {f'food_{i}_intake': float(form_data.get(f'manual_food_{i}_intake', '0')) for i in
                                  range(submitted_num_foods)}
                manual_results_data = calculate_manual_diet(original_input_data, manual_intakes)
                context['manual_results'] = manual_results_data
                context['info_message'] = "수동 입력 식단 계산 완료."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'puzzles_logic_app/diet_problem_demo.html', context)


@log_view_activity
def sports_scheduling_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sports_scheduling_introduction'
    }
    return render(request, 'puzzles_logic_app/sports_scheduling_introduction.html', context)


@log_view_activity
def sports_scheduling_demo_view(request):
    teams_list = []

    if request.method == 'GET':
        submitted_num_teams = int(request.GET.get('num_teams_to_show', preset_sport_schedule_num_teams))
        submitted_schedule_type = request.GET.get('schedule_type', preset_sport_schedule_type)
        submitted_objective = request.GET.get('objective_choice', preset_sport_schedule_objective_choice)
        submitted_max_consecutive = int(request.GET.get('max_consecutive', preset_sport_schedule_max_consecutive))
        submitted_solver_type = request.GET.get('solver_type', preset_sport_schedule_solver_type_options_list)

        for i in range(submitted_num_teams):
            team_name = request.GET.get(f'team_{i}_name', preset_sport_schedule_team_list[i])
            teams_list.append(team_name)

    elif request.method == 'POST':
        form_data = request.POST
        submitted_num_teams = int(form_data.get('num_teams', preset_sport_schedule_num_teams))
        submitted_schedule_type = form_data.get('schedule_type', preset_sport_schedule_type)
        submitted_objective = form_data.get('objective_choice', preset_sport_schedule_objective_choice)
        submitted_max_consecutive = int(form_data.get('max_consecutive', preset_sport_schedule_max_consecutive))
        submitted_solver_type = form_data.get('solver_type', preset_sport_schedule_solver_type_options_list)

        for i in range(submitted_num_teams):
            team_name = form_data.get(f'team_{i}_name', preset_sport_schedule_team_list[i])
            teams_list.append(team_name)

    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'Sports Scheduling Demo',
        'teams_list': teams_list,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_teams_options': range(2, 11),
        'submitted_num_teams': submitted_num_teams,
        'schedule_type_options': preset_sport_schedule_type_options_list,
        'submitted_schedule_type': submitted_schedule_type,
        'objective_options': preset_sport_schedule_objective_list,
        'submitted_objective': submitted_objective,
        'max_consecutive_options': range(2, 6),
        'submitted_max_consecutive': submitted_max_consecutive,
        'solver_type_options': preset_sport_schedule_solver_type_options_list,
        'submitted_solver_type': submitted_solver_type,
        'all_teams_for_matrix': preset_sport_schedule_team_list,
        'full_distance_matrix': preset_sport_schedule_dist_map_10
    }

    if request.method == 'POST':
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_sports_scheduling_json_data(form_data, submitted_num_teams,
                                                            submitted_objective, submitted_schedule_type)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_puzzle_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            if submitted_solver_type == settings.SOLVER_GUROBI:
                if submitted_num_teams <= 5:
                    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_gurobi2(input_data)
                else:
                    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_gurobi1(input_data)
            else:
                if submitted_num_teams <= 5:
                    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_ortools2(input_data)
                else:
                    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_ortools1(input_data)
                # results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_ortools1(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                success_message = (f"Total distance: {results_data['total_distance']} km, "
                                   f"Distance Gap: {results_data['distance_gap']} km,"
                                   f"Total Breaks: {results_data['total_breaks']} 번")
                if 'time_limit' in results_data and results_data['time_limit'] is not None:
                    success_message = f"{success_message}, {results_data['time_limit']}"
                context['success_message'] = success_message

        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'puzzles_logic_app/sports_scheduling_demo.html', context)


@log_view_activity
def tsp_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'tsp_introduction'
    }
    return render(request, 'puzzles_logic_app/tsp_introduction.html', context)


def tsp_demo_view(request):
    all_city_names = [city['name'] for city in preset_tsp_all_cities]
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'TSP Demo',
        'all_cities': all_city_names,
        'selected_cities': preset_tsp_cities,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'manual_results': None, 'original_input_json': None,
    }

    if request.method == 'POST':
        form_data = request.POST
        action = form_data.get('action', 'optimize')
        selected_city_names = form_data.getlist('cities')
        context['selected_cities'] = selected_city_names

        if len(selected_city_names) < 2:
            context['error_message'] = "최소 2개 이상의 도시를 선택해야 합니다."
        else:
            try:
                if action == 'optimize':
                    selected_cities_data = [city for city in preset_tsp_all_cities if city['name'] in selected_city_names]
                    # 1. 선택된 도시에 대한 부분 거리 행렬 생성
                    validation_error_msg, input_data = create_tsp_json_data(selected_cities_data)

                    # 2. 파일 저장
                    if settings.SAVE_DATA_FILE:
                        success_save_message, save_error = save_puzzle_json_data(input_data)
                        if save_error:
                            context['error_message'] = save_error
                        elif success_save_message:
                            context['success_save_message'] = success_save_message
                    if validation_error_msg:
                        context['error_message'] = validation_error_msg

                    # 2. 최적화 실행
                    results_data, error_msg, processing_time = run_tsp_optimizer(input_data)
                    context['processing_time_seconds'] = processing_time

                    if error_msg:
                        context['error_message'] = error_msg
                    elif results_data:
                        # 결과에 좌표 정보도 포함하여 템플릿으로 전달
                        tour_indices = results_data['tour_indices']
                        tour_data = [selected_cities_data[i] for i in tour_indices]

                        results_data['tour_cities_data'] = tour_data
                        results_data['tour_cities'] = " → ".join([city['name'] for city in tour_data])
                        context['results'] = results_data
                        context['success_message'] = f"최단 경로를 찾았습니다! 총 이동 거리는 {results_data['total_distance']}km 입니다."
                        # 수동 비교를 위해 원본 선택 데이터 저장
                        context['original_input_json'] = json.dumps(selected_cities_data, ensure_ascii=False)

                elif action == 'manual_check':
                    manual_tour_str = form_data.get('manual_tour', '')
                    original_input_str = form_data.get('original_input_json', '[]')

                    manual_tour_cities = [city.strip() for city in manual_tour_str.split('→') if city.strip()]
                    original_selected_cities = json.loads(original_input_str)

                    if len(manual_tour_cities) != len(original_selected_cities):
                        raise ValueError("수동 경로의 도시 수가 원래 문제와 다릅니다.")

                    # 수동 경로 거리 계산
                    manual_distance = calculate_manual_tour_distance(manual_tour_cities, preset_tsp_all_cities,
                                                                     preset_tsp_distance_matrix)

                    context['manual_results'] = {
                        'tour': ' → '.join(manual_tour_cities + ['서울']),
                        'distance': manual_distance
                    }
                    # 비교를 위해 최적 결과도 다시 context에 추가
                    context['results'] = json.loads(request.POST.get('optimal_results_json', '{}'))
                    context['original_input_json'] = original_input_str  # 계속 전달

            except Exception as e:
                context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'puzzles_logic_app/tsp_demo.html', context)


def calculate_manual_tour_distance(tour_city_names, all_cities_data, distance_matrix):
    """사용자가 입력한 순서의 도시 이름 리스트를 받아 총 거리를 계산합니다."""
    # 도시 이름을 전체 데이터의 인덱스로 변환하는 맵 생성
    name_to_idx_map = {city['name']: i for i, city in enumerate(all_cities_data)}

    # 투어 경로를 인덱스 리스트로 변환
    tour_indices = [name_to_idx_map[name] for name in tour_city_names]

    # 출발지(0)를 맨 앞과 뒤에 추가
    full_tour = [0] + tour_indices + [0]

    total_distance = 0
    for i in range(len(full_tour) - 1):
        from_node = full_tour[i]
        to_node = full_tour[i + 1]
        total_distance += distance_matrix[from_node][to_node]

    return total_distance


@log_view_activity
def sudoku_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sudoku_introduction'
    }
    return render(request, 'puzzles_logic_app/sudoku_introduction.html', context)


@log_view_activity
def sudoku_demo_view(request):
    input_grid = []
    form_data ={}
    if request.method == 'GET':
        submitted_difficulty = request.GET.get('difficulty', preset_sudoku_difficulty)
        submitted_size = int(request.GET.get('size', preset_sudoku_size))
        if submitted_size == 9:
            input_grid = generate_sudoku(submitted_difficulty, submitted_size)
        else:
            solution_grid = preset_sudoku_examples[submitted_size]
            input_grid = create_puzzle_from_solution(solution_grid, submitted_difficulty)

    elif request.method == 'POST':
        form_data = request.POST
        submitted_difficulty = form_data.get('difficulty', preset_sudoku_difficulty)
        submitted_size = int(form_data.get('size', preset_sudoku_size))
        for i in range(submitted_size):
            row = []
            for j in range(submitted_size):
                cell_value_str = form_data.get(f'cell_{i}_{j}', '0')
                # 빈 문자열이나 공백은 0으로 처리
                cell_value = int(cell_value_str) if cell_value_str.strip() else 0
                row.append(cell_value)
            input_grid.append(row)

    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'Sudoku Demo',
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'size_options': preset_sudoku_size_options,
        'submitted_size': submitted_size,
        'difficulty_options': preset_sudoku_difficulty_options,
        'submitted_difficulty': submitted_difficulty,
        'input_grid': input_grid,
        'cell_indices': range(submitted_size),
        'cell_subgrid_size': int(math.sqrt(submitted_size)),
    }

    if request.method == 'POST':
        try:
            # 1. 데이터 파일 새성 및 검증
            validation_error_msg, input_data = create_sudoku_json_data(form_data)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_puzzle_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message
            if validation_error_msg:
                context['error_message'] = validation_error_msg
            else:
                # 3. 최적화 실행
                solved_grid, error_msg_opt, processing_time = run_sudoku_solver_optimizer(input_data)
                context['processing_time_seconds'] = processing_time

                if error_msg_opt:
                    context['error_message'] = error_msg_opt
                elif solved_grid:
                    context['results'] = solved_grid
                    context['success_message'] = "스도쿠 퍼즐을 성공적으로 풀었습니다!"

        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'puzzles_logic_app/sudoku_demo.html', context)



