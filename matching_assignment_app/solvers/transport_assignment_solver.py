import logging
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger('matching_assignment_app')


class TransportAssignmentSolver(BaseOrtoolsLinearSolver):
    """
    운송 할당 문제(Transportation Assignment Problem)를 정의하고 해결하는 클래스.
    OR-Tools의 MIP 솔버(SCIP)를 사용합니다.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 운송 할당 문제 데이터를 파싱합니다.
        """
        # 부모 클래스 생성자에 input_data와 사용할 솔버 이름('SCIP')을 전달합니다.
        super().__init__(input_data, 'SCIP')

        # 이 문제에만 필요한 데이터들을 인스턴스 변수로 설정합니다.
        self.workers = self.input_data.get('driver_names', [])
        self.tasks = self.input_data.get('zone_names', [])
        self.costs = self.input_data.get('cost_matrix', [])
        self.num_workers = len(self.workers)
        self.num_tasks = len(self.tasks)

        # 결정 변수를 저장할 딕셔너리
        self.x = {}

    def _create_variables(self):
        """결정 변수 x_ij (작업자 i가 태스크 j에 할당되면 1)를 생성합니다."""
        logger.solve("--- 1. Creating Decision Variables ---")
        for i in range(self.num_workers):
            for j in range(self.num_tasks):
                self.x[i, j] = self.solver.IntVar(0, 1, f'x_{i}_{j}')
        logger.solve(f"Created {len(self.x)} assignment variables.")

    def _add_constraints(self):
        """모든 할당 제약 조건들을 모델에 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")

        # 제약 1: 각 작업자(기사)는 최대 하나의 태스크(구역)에만 할당됩니다.
        for i in range(self.num_workers):
            self.solver.Add(self.solver.Sum([self.x[i, j] for j in range(self.num_tasks)]) == 1)

        # 제약 2: 각 태스크(구역)는 정확히 하나의 작업자에게 할당됩니다.
        for j in range(self.num_tasks):
            self.solver.Add(self.solver.Sum([self.x[i, j] for i in range(self.num_workers)]) == 1)

        logger.solve(f"Added {self.num_workers + self.num_tasks} assignment constraints.")

    def _set_objective_function(self):
        """목표 함수 (총비용 최소화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")
        objective_terms = []
        for i in range(self.num_workers):
            for j in range(self.num_tasks):
                objective_terms.append(self.costs[i][j] * self.x[i, j])
        self.solver.Minimize(self.solver.Sum(objective_terms))
        logger.solve("Objective function set to minimize total assignment cost.")

    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for transport assignment...")

        assignments = []
        for i in range(self.num_workers):
            for j in range(self.num_tasks):
                if self.x[i, j].solution_value() > 0.5:
                    assignments.append({
                        'worker_name': self.workers[i],
                        'task_name': self.tasks[j],
                        'cost': self.costs[i][j]
                    })

        return {
            'assignments': assignments,
            'total_cost': self.solver.Objective().Value()
        }
