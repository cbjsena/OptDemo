import logging
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common_utils.ortools_solvers import BaseOrtoolsRoutingSolver

logger = logging.getLogger('routing_logistics_app')


class PdpSolver(BaseOrtoolsRoutingSolver):
    """
    PDP(수거 및 배송 문제)를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.depot_location = self.input_data.get('depot_location')
        self.pickup_delivery_pairs = self.input_data.get('pickup_delivery_pairs', [])
        self.num_vehicles = self.input_data.get('num_vehicles')
        self.vehicle_capacities = self.input_data.get('vehicle_capacities')

        # 모델링을 위한 위치, 수요, 픽업-배송 인덱스 리스트 생성
        self.locations = [(self.depot_location['x'], self.depot_location['y'])]
        self.demands = [0]
        self.pickup_delivery_indices = []

        node_index = 1
        for pair in self.pickup_delivery_pairs:
            # 수거 지점 추가
            self.locations.append((pair['pickup']['x'], pair['pickup']['y']))
            self.demands.append(pair['demand'])  # 수거 시 수요량은 양수
            pickup_node_index = node_index
            node_index += 1

            # 배송 지점 추가
            self.locations.append((pair['delivery']['x'], pair['delivery']['y']))
            self.demands.append(-pair['demand'])  # 배송 시 수요량은 음수
            delivery_node_index = node_index
            node_index += 1

            self.pickup_delivery_indices.append([pickup_node_index, delivery_node_index])

        self.distance_matrix = self._compute_distance_matrix()

    def _create_variables(self):
        """RoutingIndexManager와 RoutingModel 객체를 생성합니다."""
        logger.solve("--- 1. Creating Routing Model for PDP ---")
        self.manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), self.num_vehicles, 0)
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _add_constraints(self):
        """거리/수요 콜백, 용량 제약, 그리고 PDP의 핵심인 픽업-배송 제약을 추가합니다."""
        logger.solve("--- 2. Registering Callbacks, Dimensions, and PDP Constraints ---")

        # 거리 콜백 및 비용 평가자 설정
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 수요량 콜백 및 용량 차원(Dimension) 추가
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return self.demands[from_node]

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,              # null capacity slack (0이면 용량 초과 시 패널티 없음, 용량 제약 위반 불가)
            self.vehicle_capacities, # 각 차량의 용량 리스트
            True,   # start cumul to zero (차고지에서 시작 시 누적 수요 0)
            'Capacity'          # 차원의 이름
        )

        # [핵심] 픽업-배송 쌍 제약 추가 (선행 및 쌍 제약 동시 처리)
        for pickup_node, delivery_node in self.pickup_delivery_indices:
            pickup_index = self.manager.NodeToIndex(pickup_node)
            delivery_index = self.manager.NodeToIndex(delivery_node)
            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)
            # 여기서는 AddPickupAndDelivery가 기본적인 선행/쌍 제약을 처리함.
            # routing.AddDisjunction([pickup_index], penalty) # 방문하지 않을 경우 패널티 (선택)

    def _set_objective_function(self):
        """비용 평가자(Cost Evaluator) 설정이 목표 함수 역할을 합니다."""
        logger.solve("--- 3. Objective is set via ArcCostEvaluator ---")
        pass

    def _extract_results(self):
        """솔버 실행 후, 경로, 거리, 적재량 등 PDP 결과를 추출합니다."""
        logger.info("Extracting results for PDP...")
        routes = []
        capacity_dimension = self.routing.GetDimensionOrDie('Capacity')
        total_distance = self.solution.ObjectiveValue() / 100.0  # 원래 거리로 환산 (100 단위로 계산됨)

        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            route_nodes = []
            route_locations = []
            route_loads = []
            route_distance = 0

            while not self.routing.IsEnd(index):
                end_node_index = self.manager.IndexToNode(index)
                load_var = capacity_dimension.CumulVar(index)
                route_nodes.append(end_node_index)
                route_locations.append(self.locations[end_node_index])
                route_loads.append(self.solution.Value(load_var))

                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            end_node_index = self.manager.IndexToNode(index)
            load_var = capacity_dimension.CumulVar(index)
            route_nodes.append(end_node_index)
            route_locations.append(self.locations[end_node_index])
            route_loads.append(self.solution.Value(load_var))

            if len(route_nodes) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_nodes,
                    'route_locations': route_locations,
                    'distance': route_distance / 100.0,
                    'route_loads': route_loads,  # 각 지점 도착 시 적재량
                    'capacity': self.vehicle_capacities[vehicle_id]
                })

        return {
            'routes': routes,
            'total_distance': total_distance,
        }
