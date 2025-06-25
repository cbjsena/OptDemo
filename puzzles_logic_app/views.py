from django.conf import settings
from django.shortcuts import render
import json

from common_utils.run_puzzle_opt import *
from common_utils.data_utils_puzzle import *

logger = logging.getLogger(__name__)


def main_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'main_introduction'
    }
    logger.debug("Rendering Main Introduction for Puzzles & Real-World Logic.")
    return render(request, 'puzzles_logic_app/puzzles_logic_introduction.html', context)


def diet_problem_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'diet_problem_introduction'
    }
    logger.debug("Rendering Diet Problem introduction page.")
    return render(request, 'puzzles_logic_app/diet_problem_introduction.html', context)


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
        'active_submenu': 'diet_problem_demo',
        'nutrients_list': nutrients_list,
        'foods_list': foods_list,
        'results': None, 'manual_results': None, 'error_message': None, 'success_message': None, 'info_message': None,
        'processing_time_seconds': "N/A",
        'num_foods_options': range(2, 11),
        'num_nutrients_options': range(2, 6),
        'submitted_num_foods': submitted_num_foods,
        'submitted_num_nutrients': submitted_num_nutrients,
    }

    if request.method == 'POST':
        logger.info("Diet Problem Demo POST request processing.")
        action = form_data.get('action')
        try:
            input_data = create_diet_json_data(form_data)

            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_puzzle_json_data(input_data)
                if save_error:
                    context['error_message'] = save_error
                elif success_save_message:
                    context['success_save_message'] = success_save_message
            if action == "optimize":
                # 3. 최적화 실행
                results_data, error_msg_opt, processing_time = run_diet_optimizer(input_data)
                context['processing_time_seconds'] = processing_time

                if error_msg_opt:
                    context['error_message'] = error_msg_opt
                elif results_data:
                    context['results'] = results_data
                    context['success_message'] = f"최적 식단 계산 완료! 최소 비용: {results_data['total_cost']}"
            elif action == "manual_check":
                manual_intakes = {f'food_{i}_intake': float(form_data.get(f'manual_food_{i}_intake', '0')) for i in
                                  range(submitted_num_foods)}
                manual_results_data = calculate_manual_diet(input_data, manual_intakes)
                context['manual_results'] = manual_results_data
                context['info_message'] = "수동 입력 식단 계산 완료."
                # 최적화 결과도 함께 표시하기 위해 다시 실행
                results_data, _, _ = run_diet_optimizer(input_data)
                context['results'] = results_data

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"

    return render(request, 'puzzles_logic_app/diet_problem_demo.html', context)


# --- 2. Sports Scheduling ---
def sports_scheduling_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sports_scheduling_introduction'
    }
    logger.debug("Rendering Sports Scheduling introduction page.")
    return render(request, 'puzzles_logic_app/sports_scheduling_introduction.html', context)

def sports_scheduling_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sports_scheduling_demo'
    }
    logger.debug("Rendering Sports Scheduling demo page.")
    return render(request, 'puzzles_logic_app/sports_scheduling_demo.html', context)

# --- 3. Nurse Rostering Problem ---
def nurse_rostering_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'nurse_rostering_introduction'
    }
    logger.debug("Rendering Nurse Rostering introduction page.")
    return render(request, 'puzzles_logic_app/nurse_rostering_introduction.html', context)

def nurse_rostering_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'nurse_rostering_demo'
    }
    logger.debug("Rendering Nurse Rostering demo page.")
    return render(request, 'puzzles_logic_app/nurse_rostering_demo.html', context)

# --- 4. Traveling Salesman Problem (TSP) ---
def tsp_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'tsp_introduction'
    }
    logger.debug("Rendering TSP introduction page.")
    return render(request, 'puzzles_logic_app/tsp_introduction.html', context)

def tsp_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'tsp_demo'
    }
    logger.debug("Rendering TSP demo page.")
    return render(request, 'puzzles_logic_app/tsp_demo.html', context)

# --- 5. Sudoku Solver ---
def sudoku_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sudoku_introduction'
    }
    logger.debug("Rendering Sudoku introduction page.")
    return render(request, 'puzzles_logic_app/sudoku_introduction.html', context)

def sudoku_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sudoku_demo'
    }
    logger.debug("Rendering Sudoku demo page.")
    return render(request, 'puzzles_logic_app/sudoku_demo.html', context)



