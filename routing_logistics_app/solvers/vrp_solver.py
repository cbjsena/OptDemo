import logging
from math import sqrt
from ortools.constraint_solver import pywrapcp
from common_utils.ortools_solvers import BaseOrtoolsRoutingSolver

logger = logging.getLogger('routing_logistics_app')


class VrpSolver(BaseOrtoolsRoutingSolver):
    """
    VRP(차량 경로 문제)를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)

        # 데이터 파싱 및 거리 행렬 계산
        self.depot_location = self.input_data.get('depot_location')
        self.customer_locations = self.input_data.get('customer_locations')
        self.num_vehicles = self.input_data.get('num_vehicles')

        self.locations = [(self.depot_location['x'], self.depot_location['y'])] + \
                         [(loc['x'], loc['y']) for loc in self.customer_locations]
        self.distance_matrix = self._compute_distance_matrix()

    def _create_variables(self):
        """RoutingIndexManager와 RoutingModel 객체를 생성합니다."""
        logger.solve("--- 1. Creating Routing Model ---")
        self.manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), self.num_vehicles, 0)
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _add_constraints(self):
        """
        VRP에서는 제약이 주로 콜백 함수 형태로 등록됩니다.
        여기서는 거리 콜백을 등록합니다.
        """
        logger.solve("--- 2. Registering Callbacks (Constraints) ---")

        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def _set_objective_function(self):
        """
        VRP에서는 비용 평가자(Cost Evaluator) 설정이 목표 함수 역할을 합니다.
        _add_constraints에서 이미 완료되었으므로 여기서는 별도 작업이 없습니다.
        """
        logger.solve("--- 3. Objective function is set via ArcCostEvaluator ---")
        pass

    def _extract_results(self):
        """솔버 실행 후, 경로와 거리 등 결과를 추출합니다."""
        logger.info("Extracting results for VRP...")
        routes = []
        total_distance = self.solution.ObjectiveValue() / 100.0  # 거리 행렬이 100배로 계산되었으므로 나누기 100

        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            route_nodes = []
            route_locations = []
            route_distance = 0

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_nodes.append(node_index)
                route_locations.append(self.locations[node_index])

                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            # 마지막 노드(차고지) 추가
            node_index = self.manager.IndexToNode(index)
            route_nodes.append(node_index)
            route_locations.append(self.locations[node_index])

            if len(route_nodes) > 2:  # 실제 방문지가 있는 경로만 추가
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_nodes,
                    'route_locations': route_locations,
                    'distance': route_distance / 100.0,
                })

        return {
            'routes': routes,
            'total_distance': total_distance,
            'dropped_nodes': []  # 기본 VRP에서는 없음
        }
