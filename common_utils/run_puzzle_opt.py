from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
import logging

logger = logging.getLogger(__name__)


def run_diet_optimizer(input_data):
    logger.info("Running Diet Problem Optimizer.")

    foods = input_data['food_items']
    nutrients = input_data['nutrient_reqs']
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None, "오류: 선형 계획법 솔버(GLOP)를 생성할 수 없습니다.", 0.0

    # 변수 x_i: i번째 음식의 섭취량
    x = [solver.NumVar(f['min_intake'], f['max_intake'], f['name']) for f in foods]
    logger.debug(f"Created {len(x)} food variables.")

    # 제약: 각 영양소의 최소/최대 섭취량 만족
    for i in range(num_nutrients):
        constraint = solver.Constraint(nutrients[i]['min'], nutrients[i]['max'], nutrients[i]['name'])
        for j in range(num_foods):
            constraint.SetCoefficient(x[j], foods[j]['nutrients'][i])
    logger.debug(f"Added {num_nutrients} nutrient constraints.")

    # 목표 함수: 총 비용 최소화
    objective = solver.Objective()
    for i in range(num_foods):
        objective.SetCoefficient(x[i], foods[i]['cost'])
    objective.SetMinimization()
    logger.debug("Objective set to minimize total cost.")

    # 해결
    status = solver.Solve()

    # 결과 추출
    results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL:
        results['total_cost'] = solver.Objective().Value()

        for i in range(num_foods):
            intake = x[i].solution_value()
            if intake > 1e-6:  # 매우 작은 값은 무시
                results['diet_plan'].append({
                    'name': foods[i]['name'],
                    'intake': round(intake, 2),
                    'cost': round(intake * foods[i]['cost'], 2)
                })

        for i in range(num_nutrients):
            total_nutrient_intake = sum(foods[j]['nutrients'][i] * x[j].solution_value() for j in range(num_foods))
            results['nutrient_summary'].append({
                'name': nutrients[i]['name'],
                'min_req': nutrients[i]['min'],
                'max_req': nutrients[i]['max'],
                'actual_intake': round(total_nutrient_intake, 2)
            })
    else:
        error_msg = "최적 식단을 찾지 못했습니다. 제약 조건이 너무 엄격하거나(INFEASIBLE), 문제가 잘못 정의되었을 수 있습니다."

    return results, error_msg, solver.WallTime() * 1000


def calculate_manual_diet_result(input_data, manual_quantities):
    """사용자가 입력한 수동 식단의 비용과 영양 성분을 계산합니다."""
    foods = input_data.get('foods', [])
    nutrients = input_data.get('nutrients', [])
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    manual_results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    total_cost = 0

    for j in range(num_foods):
        quantity = manual_quantities[j]
        total_cost += foods[j]['cost'] * quantity
        if quantity > 0:
            manual_results['diet_plan'].append({'name': foods[j]['name'], 'quantity': quantity})
    manual_results['total_cost'] = round(total_cost, 2)

    for i in range(num_nutrients):
        total_nutrient = sum(foods[j]['nutrients'][i] * manual_quantities[j] for j in range(num_foods))
        is_ok = nutrients[i]['min'] <= total_nutrient <= nutrients[i]['max']
        manual_results['nutrient_summary'].append({
            'name': nutrients[i]['name'],
            'min': nutrients[i]['min'],
            'max': nutrients[i]['max'],
            'total': round(total_nutrient, 2),
            'is_ok': is_ok
        })

    return manual_results