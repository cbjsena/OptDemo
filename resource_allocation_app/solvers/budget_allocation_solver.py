import logging
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger(__name__)

class BudgetAllocationSolver(BaseOrtoolsLinearSolver):
    """
       예산 분배 최적화 문제를 정의하고 해결하는 클래스.
       OR-Tools의 선형 계획법 솔버를 사용합니다.
       """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 예산 분배 문제에 특화된 데이터를 파싱합니다.
        """
        # 부모 클래스 생성자에 input_data와 사용할 솔버 이름('GLOP')을 전달합니다.
        super().__init__(input_data, 'GLOP')

        # 이 문제에만 필요한 데이터들을 인스턴스 변수로 설정합니다.
        self.total_budget = self.input_data.get('total_budget')
        self.items_data = self.input_data.get('items_data', [])
        self.num_items = len(self.items_data)
        self.x = []  # 결정 변수를 저장할 리스트

    def _create_variables(self):
        """결정 변수 x_i (각 항목에 대한 투자액)를 생성합니다."""
        logger.solve("--- 1. Creating Decision Variables ---")
        infinity = self.solver.infinity()
        self.x = [self.solver.NumVar(0, infinity, f'x_{i}') for i in range(self.num_items)]
        logger.debug(f"Created {self.num_items} decision variables.")

    def _add_constraints(self):
        """모델에 모든 제약 조건들을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")
        infinity = self.solver.infinity()

        # 제약 1: 총 예산 제약
        constraint_total_budget = self.solver.Constraint(0, self.total_budget, 'total_budget_constraint')
        for i in range(self.num_items):
            constraint_total_budget.SetCoefficient(self.x[i], 1)

        # 제약 2: 개별 항목 투자 한도 제약
        for i in range(self.num_items):
            item = self.items_data[i]
            min_alloc = float(item.get('min_alloc', 0))
            max_alloc = float(item.get('max_alloc', infinity))
            self.x[i].SetBounds(min_alloc, max_alloc)

    def _set_objective_function(self):
        """목표 함수 (총 기대 수익 극대화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")
        objective = self.solver.Objective()
        for i in range(self.num_items):
            objective.SetCoefficient(self.x[i], float(self.items_data[i].get('return_coefficient', 0)))
        objective.SetMaximization()

    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for budget allocation...")

        total_maximized_return = self.solver.Objective().Value()

        allocations = []
        calculated_total_allocated = 0
        for i in range(self.num_items):
            item_data = self.items_data[i]
            allocated_val = self.x[i].solution_value()
            if abs(allocated_val) < 1e-6:
                allocated_val = 0.0

            allocations.append({
                'name': item_data.get('name', f'항목 {i + 1}'),
                'allocated_budget': round(allocated_val, 2),
                'expected_return': round(allocated_val * float(item_data.get('return_coefficient', 0)), 2),
                'min_alloc': item_data.get('min_alloc'),
                'max_alloc': item_data.get('max_alloc'),
                'return_coefficient': item_data.get('return_coefficient')
            })
            calculated_total_allocated += allocated_val

        utilization = round((calculated_total_allocated / self.total_budget) * 100, 1) if self.total_budget > 0 else 0

        return {
            'allocations': allocations,
            'total_maximized_return': total_maximized_return,
            'total_allocated_budget': calculated_total_allocated,
            'budget_utilization_percent': utilization
        }
