import logging
from gurobipy import Model, GRB, quicksum
from common_utils.gurobi_solvers import BaseGurobiSolver

logger = logging.getLogger('routing_logistics_app')


class FleetCascadingSolver0(BaseGurobiSolver):
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


class FleetCascadingSolver(BaseGurobiSolver):
    """
    시간 개념을 포함한 선단 재배치(Fleet Cascading) 문제를 해결하는 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        # 데이터 파싱
        self.vessels = self.input_data.get('vessels', [])
        self.routes = self.input_data.get('routes', [])
        self.costs = self.input_data.get('costs', {})
        self.planning_weeks = self.input_data.get('planning_weeks', 10)

        self.num_vessels = len(self.vessels)
        self.num_routes = len(self.routes)

        self.x = {}  # 결정 변수

    def _create_variables(self):
        """결정 변수 x_ijt (선박 i를 항로 j에 t주차에 투입하면 1)를 생성합니다."""
        logger.solve("--- 1. Creating Assignment Variables ---")
        for i in range(self.num_vessels):
            vessel = self.vessels[i]
            for j in range(self.num_routes):
                route = self.routes[j]
                for t in range(self.planning_weeks):
                    # 제약 조건에 맞는 경우에만 변수 생성
                    if vessel['type'] in route['acceptable_types'] and t + 1 >= vessel['available_week']:
                        if vessel['status'] == 'Dry Dock' and vessel['available_week'] <= t + 1 < vessel[
                            'available_week'] + 2:
                            continue
                        self.x[i, j, t] = self.model.addVar(vtype=GRB.BINARY, name=f'assign_v{i}_r{j}_t{t}')

    def _add_constraints(self):
        """제약 조건을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")

        # 제약 1: 각 선박은 최대 하나의 신규/개편 항로에만 배정됩니다.
        for i in range(self.num_vessels):
            self.model.addConstr(quicksum(
                self.x.get((i, j, t), 0) for j in range(self.num_routes) for t in range(self.planning_weeks)) <= 1)

        # 제약 2: 각 신규/개편 항로에 필요한 선박 수가 정확히 충족되어야 합니다.
        for j in range(self.num_routes):
            self.model.addConstr(quicksum(
                self.x.get((i, j, t), 0) for i in range(self.num_vessels) for t in range(self.planning_weeks)) ==
                                 self.routes[j]['required_vessels'])

        # 제약 3: 신규 항로는 특정 주차 이후에만 선박을 받을 수 있습니다.
        for j in range(self.num_routes):
            route = self.routes[j]
            for t in range(route['phase_in_week'] - 1):
                self.model.addConstr(quicksum(self.x.get((i, j, t), 0) for i in range(self.num_vessels)) == 0)

    def _set_objective_function(self):
        """목표 함수 (총 재배치 비용 최소화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")

        transition_costs = quicksum(
            self.costs['transition'].get(self.vessels[i]['available_port'], 999) * self.x.get((i, j, t), 0)
            for i, j, t in self.x.keys()
        )

        idling_costs = quicksum(
            (t + 1 - self.vessels[i]['available_week']) * self.costs['idling'] * self.x.get((i, j, t), 0)
            for i, j, t in self.x.keys()
        )

        opportunity_costs = quicksum(
            self.costs['opportunity'] * self.x.get((i, j, t), 0)
            for i, j, t in self.x.keys() if self.vessels[i]['contract'] == 'Owned'
        )

        transhipment_costs = quicksum(
            self.costs['transhipment'] * self.x.get((i, j, t), 0)
            for i, j, t in self.x.keys() if self.vessels[i]['status'] == 'Active'
        )

        self.model.setObjective(transition_costs + idling_costs + opportunity_costs + transhipment_costs, GRB.MINIMIZE)

    def _extract_results(self):
        """솔버 실행 후, 최적 배치 결과를 추출합니다."""
        logger.info("Extracting results for Fleet Cascading...")

        gantt_data = {'labels': [v['id'] for v in self.vessels], 'datasets': []}

        # 데이터셋 초기화
        datasets = {
            'Existing': {'label': 'Existing Route', 'backgroundColor': 'rgba(200, 200, 200, 0.6)', 'data': []},
            'Idling': {'label': 'Idling', 'backgroundColor': 'rgba(255, 206, 86, 0.6)', 'data': []},
            'Dry Dock': {'label': 'Dry Dock', 'backgroundColor': 'rgba(255, 99, 132, 0.6)', 'data': []}
        }
        route_colors = ['rgba(54, 162, 235, 0.6)', 'rgba(75, 192, 192, 0.6)', 'rgba(153, 102, 255, 0.6)',
                        'rgba(255, 159, 64, 0.6)', 'rgba(99, 255, 132, 0.6)']
        for i, r in enumerate(self.routes):
            datasets[r['id']] = {'label': r['id'], 'backgroundColor': route_colors[i % len(route_colors)], 'data': []}

        for i in range(self.num_vessels):
            vessel = self.vessels[i]
            assigned = False
            # 기존 항로 기간
            datasets['Existing']['data'].append([0, vessel['available_week']])

            # Dry Dock 기간
            if vessel['status'] == 'Dry Dock':
                datasets['Dry Dock']['data'].append([vessel['available_week'], vessel['available_week'] + 2])

            # 신규 항로 배정
            for j in range(self.num_routes):
                for t in range(self.planning_weeks):
                    if self.x.get((i, j, t)) and self.x[i, j, t].X > 0.5:
                        # 유휴 기간
                        if t + 1 > vessel['available_week']:
                            datasets['Idling']['data'].append([vessel['available_week'], t + 1])
                        # 신규 항로 기간
                        datasets[self.routes[j]['id']]['data'].append([t + 1, self.planning_weeks])
                        assigned = True
                        break
                if assigned: break

            # 배정되지 않은 경우, 남은 기간은 유휴
            if not assigned:
                start_idle = vessel['available_week']
                if vessel['status'] == 'Dry Dock':
                    start_idle += 2
                datasets['Idling']['data'].append([start_idle, self.planning_weeks])

        gantt_data['datasets'] = list(datasets.values())

        return {
            'total_cost': self.model.ObjVal,
            'gantt_data': gantt_data
        }