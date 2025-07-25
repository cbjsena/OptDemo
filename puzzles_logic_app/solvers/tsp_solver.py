import logging
from math import sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common_utils.ortools_solvers import BaseOrtoolsRoutingSolver

logger = logging.getLogger('puzzles_logic_app')


class TspSolver(BaseOrtoolsRoutingSolver):
    """
    TSP(외판원 문제)를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsRoutingSolver를 초기화하고 TSP 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.distance_matrix = self.input_data.get('sub_matrix')
        self.num_nodes = self.input_data.get('num_cities')
        self.num_vehicles = 1  # TSP는 차량이 1대인 VRP와 동일합니다.
        self.depot_index = 0

    def _create_variables(self):
        """RoutingIndexManager와 RoutingModel 객체를 생성합니다."""
        logger.solve("--- 1. Creating Routing Model for TSP ---")
        self.manager = pywrapcp.RoutingIndexManager(self.num_nodes, self.num_vehicles, self.depot_index)
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _add_constraints(self):
        """TSP 문제에서는 추가적인 제약 조건이 거의 필요 없으며, 주로 콜백 등록이 이루어집니다."""
        logger.solve("--- 2. Registering Transit Callback for TSP ---")
        # 이 메서드는 Base 클래스의 solve 로직에 의해 호출되므로 빈 상태로 둘 수 없습니다.
        # 거리 콜백 등록은 목표 함수 설정의 일부로 볼 수 있으므로 _set_objective_function으로 이동 가능합니다.
        pass

    def _set_objective_function(self):
        """거리 콜백을 등록하고 비용 평가자를 설정하여 목표 함수를 정의합니다."""
        logger.solve("--- 3. Setting Arc Cost Evaluator for TSP ---")

        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def _extract_results(self):
        """솔버 실행 후, 최적 경로와 총 거리를 추출합니다."""
        logger.info("Extracting results for TSP...")

        index = self.routing.Start(0)
        tour_indices = []
        while not self.routing.IsEnd(index):
            node_index = self.manager.IndexToNode(index)
            tour_indices.append(node_index)
            index = self.solution.Value(self.routing.NextVar(index))

        # 마지막으로 출발지로 돌아오는 경로 추가
        tour_indices.append(self.manager.IndexToNode(index))

        return {
            'tour_indices': tour_indices,
            'total_distance': self.solution.ObjectiveValue()
        }
