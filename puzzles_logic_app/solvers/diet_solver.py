import logging
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger('puzzles_logic_app')


class DietSolver(BaseOrtoolsLinearSolver):
    """
    다이어트 문제를 정의하고 해결하는 클래스.
    OR-Tools의 선형 계획법 솔버(GLOP)를 사용합니다.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 다이어트 문제 데이터를 파싱합니다.
        """
        super().__init__(input_data, 'GLOP')

        # 입력 데이터 파싱
        self.foods = self.input_data.get('food_items', [])
        self.nutrients = self.input_data.get('nutrient_reqs', [])
        self.num_foods = len(self.foods)
        self.num_nutrients = len(self.nutrients)

        self.all_nutrients = range(self.num_nutrients)
        self.all_foods = range(self.num_foods)
        # 결정 변수 초기화
        self.x = []  # 각 음식의 섭취량을 저장할 변수

    def _create_variables(self):
        """결정 변수 x_j (음식 j의 섭취량)를 생성합니다."""
        logger.solve("--- 1. Creating Decision Variables ---")
        self.x = [self.solver.NumVar(f['min_intake'], f['max_intake'], f['name']) for f in self.foods]
        logger.debug(f"Created {len(self.x)} food variables.")

    def _add_constraints(self):
        """각 영양소의 최소/최대 섭취량 제약 조건을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")
        for i in self.all_nutrients:
            nutrient = self.nutrients[i]
            constraint = self.solver.Constraint(nutrient['min'], nutrient['max'], nutrient['name'])
            for j in self.all_foods:
                constraint.SetCoefficient(self.x[j], self.foods[j]['nutrients'][i])
        logger.debug(f"Added {self.num_nutrients} nutrient constraints.")

    def _set_objective_function(self):
        """목표 함수 (총비용 최소화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")
        objective = self.solver.Objective()
        for j in self.all_foods:
            objective.SetCoefficient(self.x[j], self.foods[j]['cost'])
        objective.SetMinimization()
        logger.debug("Objective set to minimize total cost.")

    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for diet problem...")

        diet_plan = []
        for j in self.all_foods:
            intake = self.x[j].solution_value()
            if intake > 1e-6:  # 매우 작은 값은 무시
                diet_plan.append({
                    'name': self.foods[j]['name'],
                    'intake': round(intake, 2),
                    'cost': round(intake * self.foods[j]['cost'], 2)
                })

        nutrient_summary = []
        for i in self.all_nutrients:
            total_intake = sum(
                self.foods[j]['nutrients'][i] * self.x[j].solution_value() for j in self.all_foods)
            nutrient_summary.append({
                'name': self.nutrients[i]['name'],
                'min_req': self.nutrients[i]['min'],
                'max_req': self.nutrients[i]['max'],
                'actual_intake': round(total_intake, 2)
            })

        return {
            'diet_plan': diet_plan,
            'total_cost': self.solver.Objective().Value(),
            'nutrient_summary': nutrient_summary
        }


def calculate_manual_diet_result(input_data, manual_quantities):
    """사용자가 입력한 수동 식단의 비용과 영양 성분을 계산합니다."""
    foods = input_data.get('foods', [])
    nutrients = input_data.get('nutrients', [])
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    manual_results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    total_cost = 0

    for food in range(num_foods):
        quantity = manual_quantities[food]
        total_cost += foods[food]['cost'] * quantity
        if quantity > 0:
            manual_results['diet_plan'].append({'name': foods[food]['name'], 'quantity': quantity})
    manual_results['total_cost'] = round(total_cost, 2)

    for nut in range(num_nutrients):
        total_nutrient = sum(foods[food]['nutrients'][nut] * manual_quantities[food] for food in range(num_foods))
        is_ok = nutrients[nut]['min'] <= total_nutrient <= nutrients[nut]['max']
        manual_results['nutrient_summary'].append({
            'name': nutrients[nut]['name'],
            'min': nutrients[nut]['min'],
            'max': nutrients[nut]['max'],
            'total': round(total_nutrient, 2),
            'is_ok': is_ok
        })

    return manual_results