import logging
from gurobipy import Model, GRB, quicksum
from common_utils.gurobi_solvers import BaseGurobiSolver

logger = logging.getLogger('routing_logistics_app')


class FleetCascadingSolver(BaseGurobiSolver):
    """
    선단 재배치(Fleet Cascading) 문제를 해결하는 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        # 데이터 파싱
        self.vessels = self.input_data.get('vessels', [])
        self.routes = self.input_data.get('routes', [])
        self.transition_costs = self.input_data.get('transition_costs', {})

        self.num_vessels = len(self.vessels)
        self.num_routes = len(self.routes)

        self.x = {}  # 결정 변수

    def _create_variables(self):
        """결정 변수 x_ij (선박 i를 항로 j에 배정하면 1)를 생성합니다."""
        logger.solve("--- 1. Creating Assignment Variables ---")
        for i in range(self.num_vessels):
            for j in range(self.num_routes):
                self.x[i, j] = self.model.addVar(vtype=GRB.BINARY,
                                                 name=f'assign_{self.vessels[i]["id"]}_to_{self.routes[j]["id"]}')

    def _add_constraints(self):
        """제약 조건을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")

        # 제약 1: 각 선박은 정확히 하나의 신규 항로에만 배정됩니다.
        for i in range(self.num_vessels):
            self.model.addConstr(quicksum(self.x[i, j] for j in range(self.num_routes)) == 1, name=f"vessel_assign_{i}")

        # 제약 2: 각 신규 항로에 필요한 선박 수가 정확히 충족되어야 합니다.
        for j in range(self.num_routes):
            self.model.addConstr(
                quicksum(self.x[i, j] for i in range(self.num_vessels)) == self.routes[j]['required_vessels'],
                name=f"route_demand_{j}")

        # 제약 3: 각 항로에는 적합한 유형의 선박만 배정될 수 있습니다.
        for i in range(self.num_vessels):
            for j in range(self.num_routes):
                if self.vessels[i]['type'] not in self.routes[j]['acceptable_types']:
                    self.model.addConstr(self.x[i, j] == 0, name=f"type_mismatch_{i}_{j}")

    def _set_objective_function(self):
        """목표 함수 (총 재배치 비용 최소화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")

        objective_expr = quicksum(
            self.transition_costs.get(self.vessels[i]['current_route'], {}).get(self.routes[j]['id'], 999999) * self.x[
                i, j]
            for i in range(self.num_vessels)
            for j in range(self.num_routes)
        )
        self.model.setObjective(objective_expr, GRB.MINIMIZE)

    def _extract_results(self):
        """솔버 실행 후, 최적 배치 결과를 추출합니다."""
        logger.info("Extracting results for Fleet Cascading...")

        assignments = []
        for j in range(self.num_routes):
            route_assignment = {
                'route_id': self.routes[j]['id'],
                'assigned_vessels': []
            }
            for i in range(self.num_vessels):
                if self.x[i, j].X > 0.5:
                    route_assignment['assigned_vessels'].append(self.vessels[i]['id'])
            assignments.append(route_assignment)

        return {
            'assignments': assignments,
            'total_cost': self.model.ObjVal
        }
