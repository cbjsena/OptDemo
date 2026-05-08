import logging
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger(__name__)


class VesselDeploymentSolver(BaseOrtoolsLinearSolver):
    """
    선박 배치(Vessel Deployment) 최적화 문제를 해결하는 클래스.
    OR-Tools의 MIP 솔버(CBC)를 사용합니다.

    목적: 모든 Trade의 수요를 충족시키면서 사용하는 선박 수를 최소화
    제약:
      1. Trade별 수요 충족 (해당 Trade 내 모든 항로의 총 용량 >= Trade 수요)
      2. 선박 크기별 가용 수량 제약
      3. 항로별 운행 선박 수 제한
    """

    def __init__(self, input_data):
        super().__init__(input_data, 'CBC')

        self.vessel_sizes = self.input_data.get('vessel_sizes', [])
        self.vessel_availability = self.input_data.get('vessel_availability', {})
        self.trades = self.input_data.get('trades', [])

        # trades에서 routes를 자동 생성
        self.routes = []
        self.trade_route_indices = {}  # trade_code -> [route indices]
        idx = 0
        for trade in self.trades:
            code = trade['code']
            num_routes = trade['num_routes']
            max_vessels = trade.get('max_vessels_per_route', 999)
            self.trade_route_indices[code] = []
            for i in range(num_routes):
                self.routes.append({
                    'name': f"{code}{i + 1}",
                    'trade': code,
                    'max_vessels': max_vessels,
                })
                self.trade_route_indices[code].append(idx)
                idx += 1

        self.num_routes = len(self.routes)
        self.num_sizes = len(self.vessel_sizes)
        self.x = {}

    def _create_variables(self):
        """결정 변수 생성: x[r][s] = 항로 r에 크기 s 선박 배치 수"""
        logger.info("--- 1. Creating Decision Variables ---")

        for r in range(self.num_routes):
            route = self.routes[r]
            for s in range(self.num_sizes):
                size = self.vessel_sizes[s]
                avail = self.vessel_availability.get(str(size), 0)
                var_name = f'x_{route["name"]}_{size}'
                self.x[r, s] = self.solver.IntVar(0, avail, var_name)

        logger.info(f"Created {len(self.x)} decision variables "
                    f"({self.num_routes} routes x {self.num_sizes} vessel sizes)")

    def _add_constraints(self):
        """제약 조건 추가"""
        logger.info("--- 2. Adding Constraints ---")

        # 제약 1: Trade별 수요 충족
        for trade in self.trades:
            code = trade['code']
            demand = trade['demand']
            route_indices = self.trade_route_indices[code]
            constraint_expr = sum(
                self.x[r, s] * self.vessel_sizes[s]
                for r in route_indices
                for s in range(self.num_sizes)
            )
            self.solver.Add(constraint_expr >= demand)
            logger.info(f"  Trade demand: {code} (routes: {len(route_indices)}) >= {demand} TEU")

        # 제약 2: 선박 크기별 가용 수량 제약
        for s in range(self.num_sizes):
            size = self.vessel_sizes[s]
            avail = self.vessel_availability.get(str(size), 0)
            constraint_expr = sum(self.x[r, s] for r in range(self.num_routes))
            self.solver.Add(constraint_expr <= avail)
            logger.info(f"  Fleet constraint: size {size} <= {avail}")

        # 제약 3: 항로별 운행 선박 수 제한
        for r in range(self.num_routes):
            route = self.routes[r]
            max_v = route.get('max_vessels', 999)
            if max_v < 999:
                constraint_expr = sum(self.x[r, s] for s in range(self.num_sizes))
                self.solver.Add(constraint_expr <= max_v)
                logger.info(f"  Route vessel limit: {route['name']} <= {max_v}")

    def _set_objective_function(self):
        """목적 함수: 총 선박 수 최소화"""
        logger.info("--- 3. Setting Objective Function ---")

        total_vessels = sum(
            self.x[r, s] for r in range(self.num_routes) for s in range(self.num_sizes)
        )
        self.solver.Minimize(total_vessels)
        logger.info("Objective: Minimize total number of vessels deployed")

    def _extract_results(self):
        """최적화 결과 추출"""
        logger.info("--- 4. Extracting Results ---")

        deployment_matrix = []
        total_vessels_used = 0

        for r in range(self.num_routes):
            route = self.routes[r]
            row = {
                'name': route['name'],
                'trade': route.get('trade', ''),
                'deployment': [],
                'total_vessels': 0,
                'total_capacity': 0,
            }
            for s in range(self.num_sizes):
                count = int(self.x[r, s].solution_value())
                row['deployment'].append(count)
                row['total_vessels'] += count
                row['total_capacity'] += count * self.vessel_sizes[s]

            total_vessels_used += row['total_vessels']
            deployment_matrix.append(row)

        # 선박 크기별 사용량 vs 가용량
        fleet_usage = []
        for s in range(self.num_sizes):
            size = self.vessel_sizes[s]
            used = sum(int(self.x[r, s].solution_value()) for r in range(self.num_routes))
            avail = self.vessel_availability.get(str(size), 0)
            fleet_usage.append({
                'size': size,
                'used': used,
                'available': avail,
                'utilization': round(used / avail * 100, 1) if avail > 0 else 0,
            })

        # Trade 그룹별 요약
        trade_summary = []
        for trade in self.trades:
            code = trade['code']
            route_indices = self.trade_route_indices[code]
            vessels = sum(deployment_matrix[r]['total_vessels'] for r in route_indices)
            capacity = sum(deployment_matrix[r]['total_capacity'] for r in route_indices)
            trade_summary.append({
                'code': code,
                'demand': trade['demand'],
                'num_routes': trade['num_routes'],
                'vessels': vessels,
                'capacity': capacity,
                'surplus': capacity - trade['demand'],
            })

        results = {
            'deployment_matrix': deployment_matrix,
            'fleet_usage': fleet_usage,
            'trade_summary': trade_summary,
            'total_vessels_used': total_vessels_used,
            'vessel_sizes': self.vessel_sizes,
            'objective_value': self.solver.Objective().Value(),
        }

        logger.info(f"Total vessels deployed: {total_vessels_used}")
        logger.info(f"Objective value: {self.solver.Objective().Value()}")

        return results

