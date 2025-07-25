import logging
from math import sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common_utils.ortools_solvers import BaseOrtoolsRoutingSolver

logger = logging.getLogger('routing_logistics_app')


class CvrpSolver(BaseOrtoolsRoutingSolver):
    """
    CVRP(용량 제약 차량 경로 문제)를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)

        # 데이터 파싱
        self.depot_location = self.input_data.get('depot_location')
        self.customer_locations = self.input_data.get('customer_locations')
        self.num_vehicles = self.input_data.get('num_vehicles')
        self.vehicle_capacities = self.input_data.get('vehicle_capacities')

        self.locations = [(self.depot_location['x'], self.depot_location['y'])] + \
                         [(loc['x'], loc['y']) for loc in self.customer_locations]
        self.demands = [0] + [loc.get('demand', 0) for loc in self.customer_locations]
        self.distance_matrix = self._compute_distance_matrix()

    def _create_variables(self):
        """RoutingIndexManager와 RoutingModel 객체를 생성합니다."""
        logger.solve("--- 1. Creating Routing Model ---")
        self.manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), self.num_vehicles, 0)
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _add_constraints(self):
        """거리 및 수요 콜백을 등록하고, 용량 제약(Dimension)을 추가합니다."""
        logger.solve("--- 2. Registering Callbacks & Dimensions ---")

        # 거리 콜백
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 수요량 콜백 및 용량 차원 추가
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return self.demands[from_node]

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.vehicle_capacities,
            True,  # start cumul to zero
            'Capacity'
        )

    def _set_objective_function(self):
        """비용 평가자(Cost Evaluator) 설정이 목표 함수 역할을 합니다."""
        logger.solve("--- 3. Objective is set via ArcCostEvaluator ---")
        pass

    def _extract_results(self):
        """솔버 실행 후, 경로, 거리, 적재량 등 결과를 추출합니다."""
        logger.info("Extracting results for CVRP...")
        routes = []
        total_distance = self.solution.ObjectiveValue() / 100.0  # 거리 행렬이 정수로 되어 있으므로 100으로 나누어 실제 거리로 변환
        total_demand_served = 0

        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            route_nodes = []
            route_locations = []
            route_distance = 0
            route_load = 0

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_nodes.append(node_index)
                route_locations.append(self.locations[node_index])
                route_load += self.demands[node_index]

                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            node_index = self.manager.IndexToNode(index)
            route_nodes.append(node_index)
            route_locations.append(self.locations[node_index])

            if len(route_nodes) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_nodes,
                    'route_locations': route_locations,
                    'distance': route_distance / 100.0,
                    'load': route_load,
                    'capacity': self.vehicle_capacities[vehicle_id]
                })
                total_demand_served += route_load

        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_demand_served': total_demand_served,
            'dropped_nodes': []
        }
