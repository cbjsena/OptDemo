from common_utils.common_data_utils import save_json_data
import logging
import datetime

from django.conf import settings

logger = logging.getLogger(__name__)

preset_diet_nutrient_number = 2
preset_diet_food_number = 4
preset_diet_nutrients_data = [
    {'name': '칼로리(kcal)', 'min': '100', 'max': '2500'},
    {'name': '단백질(g)', 'min': '20', 'max': '100'},
    {'name': '지방(g)', 'min': '20', 'max': '70'},
    {'name': '탄수화물(g)', 'min': '20', 'max': '350'},
    {'name': '나트륨(mg)', 'min': '0', 'max': '2000'}
]
preset_diet_foods_data = [
    {'name': '우유(100ml)', 'cost': '150', 'min_intake': '2', 'max_intake': '10',
     'nutrients': ['60', '3.2', '3.5', '4.8', '50']},
    {'name': '계란(1개)', 'cost': '300', 'min_intake': '0', 'max_intake': '5', 'nutrients': ['80', '6', '6', '0.5', '65']},
    {'name': '식빵(1장)', 'cost': '200', 'min_intake': '0', 'max_intake': '10',
     'nutrients': ['70', '2.5', '1', '13', '150']},
    {'name': '닭가슴살(100g)', 'cost': '1500', 'min_intake': '2', 'max_intake': '5',
     'nutrients': ['110', '23', '1.5', '0', '70']},
    {'name': '바나나(1개)', 'cost': '500', 'min_intake': '0', 'max_intake': '4',
     'nutrients': ['90', '1', '0.3', '23', '1']},
    {'name': '아몬드(10g)', 'cost': '200', 'min_intake': '0', 'max_intake': '5', 'nutrients': ['60', '2', '5', '2', '1']},
    {'name': '두부(100g)', 'cost': '500', 'min_intake': '0', 'max_intake': '3',
     'nutrients': ['80', '8', '4.5', '2', '5']},
    {'name': '현미밥(100g)', 'cost': '400', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['130', '2.5', '1', '28', '3']},
    {'name': '시금치(100g)', 'cost': '300', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['25', '2.9', '0.4', '3.6', '80']},
    {'name': '올리브 오일(10g)', 'cost': '100', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['90', '0', '10', '0', '0']},
]
preset_sport_schedule_max_consecutive=3
preset_sport_schedule_objective_choice='minimize_travel'
preset_sport_schedule_type='double'
preset_sport_schedule_type_options_list = [
    {'value': 'single', 'name': '싱글 라운드 로빈 (팀당 1경기)'},
    {'value': 'double', 'name': '더블 라운드 로빈 (팀당 2경기)'},
]
preset_sport_schedule_objective_list = [
    {'value': 'minimize_travel', 'name': '총 이동 거리 최소화'},
    {'value': 'fairness', 'name': '연속 홈/원정 최소화'},
    {'value': 'distance_gap', 'name': '팀간 이동거리 차이 최소화'},
]
preset_sport_schedule_solver_type_options_list = [
    {'value': settings.SOLVER_ORTOOLS, 'name': 'OR-Tools'},
    {'value': settings.SOLVER_GUROBI, 'name': 'Gurobi'},
]

preset_sport_schedule_num_teams = 4
preset_sport_schedule_team_list = [
    "한화", "LG", "롯데", "KIA", "삼성",
    "KT", "SSG", "NC", "두산", "키움"
]

# 각 팀의 연고지 도시 (거리 계산용, 순서는 위와 동일)
preset_sport_schedule_cities = ['대전', '서울', '부산', '광주', '대구', '수원', '인천', '창원', '서울', '서울']
# 도시 간 대략적인 거리 행렬 (km) - 예시 데이터
preset_sport_schedule_distance_matrix_km = [
    # 대전, 서울, 부산, 광주, 대구, 수원, 인천, 창원
    [  0, 160, 200, 140, 100, 130, 200, 150], # 대전
    [160,   0, 325, 270, 240,  30,  30, 290], # 서울
    [200, 325,   0, 200,  95, 300, 350,  40], # 부산
    [140, 270, 200,   0, 150, 240, 300, 160], # 광주
    [100, 240,  95, 150,   0, 210, 265,  60], # 대구
    [130,  30, 300, 240, 210,   0,  40, 260], # 수원
    [200,  30, 350, 300, 265,  40,   0, 320], # 인천
    [150, 290,  40, 160,  60, 260, 320,   0], # 창원
]

preset_sport_schedule_dist_map_10 = [[0] * 10 for _ in range(10)]
preset_sport_schedule_city_map = [0, 1, 2, 3, 4, 5, 6, 7, 1, 1]  # 10개 팀의 도시 인덱스
for i in range(10):
    for j in range(10):
        city_i = preset_sport_schedule_city_map[i]
        city_j = preset_sport_schedule_city_map[j]
        if city_i < 8 and city_j < 8:
            preset_sport_schedule_dist_map_10[i][j] = preset_sport_schedule_distance_matrix_km[city_i][city_j]

def create_diet_json_data(form_data):
    logger.debug("Creating and validating Diet Problem input data from form.")
    num_foods = int(form_data.get('num_foods', 0))
    num_nutrients = int(form_data.get('num_nutrients', 0))

    nutrient_reqs=[]
    # 1. 영양소 요구사항 파싱
    for i in range(num_nutrients):
        try:
            nutrient_reqs.append({
                'name': form_data.get(f'nutrient_{i}_name'),
                'min': float(form_data.get(f'nutrient_{i}_min')),
                'max': float(form_data.get(f'nutrient_{i}_max'))
            })
        except (ValueError, TypeError):
            raise ValueError(f"영양소 {i + 1}의 최소/최대 요구량 값이 올바른 숫자가 아닙니다.")

    # 2. 식품 데이터 파싱
    food_items=[]
    for i in range(num_foods):
        try:
            food_item = {
                'name': form_data.get(f'food_{i}_name'),
                'cost': float(form_data.get(f'food_{i}_cost')),
                'min_intake': float(form_data.get(f'food_{i}_min_intake', 0)),
                'max_intake': float(form_data.get(f'food_{i}_max_intake', 10000)),  # 충분히 큰 값
                'nutrients': []
            }
            for j in range(num_nutrients):
                nutrient_val = float(form_data.get(f'nutrient_val_{i}_{j}'))
                food_item['nutrients'].append(nutrient_val)
            food_items.append(food_item)
        except (ValueError, TypeError):
            raise ValueError(f"식품 '{form_data.get(f'food_{i}_name')}'의 입력값이 올바르지 않습니다.")

    input_data = {
        "problem_type": "diet_problem",
        "num_foods": num_foods,
        "num_nutrients": num_nutrients,
        "food_items": food_items,
        "nutrient_reqs": nutrient_reqs
    }

    logger.info("End Diet Problem Demo Input data processing.")
    return input_data


def calculate_manual_diet(input_data, manual_intakes):
    """수동 입력 식단의 비용과 영양 정보를 계산합니다."""
    logger.info("Calculating manual diet plan.")

    foods = input_data['food_items']
    nutrients = input_data['nutrient_reqs']
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    manual_results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    total_cost = 0.0

    for i in range(num_foods):
        intake = manual_intakes.get(f'food_{i}_intake', 0)
        if intake > 0:
            food_item = foods[i]
            cost = intake * food_item['cost']
            total_cost += cost
            manual_results['diet_plan'].append({
                'name': food_item['name'],
                'intake': intake,
                'cost': round(cost, 2)
            })

    manual_results['total_cost'] = round(total_cost, 2)

    for i in range(num_nutrients):
        total_nutrient_intake = 0
        for j in range(num_foods):
            intake = manual_intakes.get(f'food_{j}_intake', 0)
            total_nutrient_intake += foods[j]['nutrients'][i] * intake

        reqs = nutrients[i]
        status = "OK"
        if total_nutrient_intake < reqs['min']:
            status = "Minimum not met"
        elif total_nutrient_intake > reqs['max']:
            status = "Maximum exceeded"

        manual_results['nutrient_summary'].append({
            'name': reqs['name'],
            'min_req': reqs['min'],
            'max_req': reqs['max'],
            'actual_intake': round(total_nutrient_intake, 2),
            'status': status
        })

    return manual_results


def create_sports_scheduling_json_data(form_data, num_teams, objective, schedule_type):
    # 팀 이름 리스트 생성
    teams_list = [form_data.get(f'team_{i}_name') for i in range(num_teams)]
    # 선택된 팀에 해당하는 거리 행렬 슬라이싱
    selected_dist_matrix = [[0] * num_teams for _ in range(num_teams)]
    for i in range(num_teams):
        for j in range(num_teams):
            # default_teams_pool에서의 원래 인덱스를 찾아야 함
            original_idx_i = preset_sport_schedule_team_list.index(teams_list[i]) if teams_list[i] in preset_sport_schedule_team_list else -1
            original_idx_j = preset_sport_schedule_team_list.index(teams_list[j]) if teams_list[j] in preset_sport_schedule_team_list else -1
            if original_idx_i != -1 and original_idx_j != -1:
                selected_dist_matrix[i][j] = preset_sport_schedule_dist_map_10[original_idx_i][original_idx_j]

    input_data = {
        'problem_type': 'sports_scheduling',
        'teams': teams_list,
        'num_teams': num_teams,
        'distance_matrix': selected_dist_matrix,  # 거리 행렬 추가
        'objective_choice': objective,
        'schedule_type': schedule_type,
        'max_consecutive': int(form_data.get('max_consecutive'))
    }

    return input_data


def save_puzzle_json_data(input_data):
    problem_type = input_data.get('problem_type')
    dir = f'puzzles_{problem_type}_data'
    filename_pattern = ''

    if "diet_problem" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"
    elif "sports_scheduling" == problem_type:
        num_teams = input_data.get('num_teams')
        objective_choice = input_data.get('objective_choice')
        schedule_type = input_data.get('schedule_type')
        filename_pattern = f"{objective_choice}_{schedule_type}_team{num_teams}"
    elif "nurse_rostering" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"
    elif "tsp" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"
    elif "sudoku" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"

    return save_json_data(input_data, dir, filename_pattern)