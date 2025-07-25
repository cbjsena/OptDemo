import logging
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger('matching_assignment_app')


def create_cost_matrix(cf_panels, tft_panels):
    logger.debug("Calculating yield matrix...")
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)
    yield_matrix = [[-1] * num_tft for _ in range(num_cf)]
    for i in range(num_cf):
        cf_panel = cf_panels[i]
        cf_map = cf_panel.get('defect_map')
        cf_rows = cf_panel.get('rows')
        cf_cols = cf_panel.get('cols')

        # 필수 키 누락 또는 유효하지 않은 값(None 또는 0) 확인
        if not all([isinstance(cf_map, list), isinstance(cf_rows, int), cf_rows > 0, isinstance(cf_cols, int),
                    cf_cols > 0]):
            logger.warning(f"CF Panel {cf_panel.get('id', i)} has invalid structure or dimensions. Skipping.")
            continue  # 이 CF 패널은 모든 TFT와 매칭 불가 (-1 유지)

        for j in range(num_tft):
            tft_panel = tft_panels[j]
            tft_map = tft_panel.get('defect_map')
            tft_rows = tft_panel.get('rows')
            tft_cols = tft_panel.get('cols')

            if not all([isinstance(tft_map, list), isinstance(tft_rows, int), tft_rows > 0, isinstance(tft_cols, int),
                        tft_cols > 0]):
                logger.warning(
                    f"TFT Panel {tft_panel.get('id', j)} has invalid structure or dimensions. Marking as unmatchable with CF {cf_panel.get('id', i)}.")
                # yield_matrix[i][j]는 이미 -1
                continue

            if cf_rows != tft_rows or cf_cols != tft_cols:
                logger.debug(
                    f"Dimension mismatch between CF {cf_panel.get('id', i)} ({cf_rows}x{cf_cols}) and TFT {tft_panel.get('id', j)} ({tft_rows}x{tft_cols}).")
                # yield_matrix[i][j]는 이미 -1
                continue

            current_yield = 0
            valid_cell_structure = True
            if len(cf_map) != cf_rows or len(tft_map) != tft_rows:  # defect_map의 행 개수 확인
                logger.warning(
                    f"Defect map row count mismatch for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                valid_cell_structure = False

            if valid_cell_structure:
                for r in range(cf_rows):
                    if not valid_cell_structure: break
                    # 각 행의 열 개수 및 셀 값 유효성 확인
                    if len(cf_map[r]) != cf_cols or len(tft_map[r]) != tft_cols:
                        logger.warning(
                            f"Defect map col count mismatch at row {r} for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                        valid_cell_structure = False
                        break

                    for c in range(cf_cols):
                        cf_cell = cf_map[r][c]
                        tft_cell = tft_map[r][c]
                        if not (cf_cell in (0, 1) and tft_cell in (0, 1)):
                            logger.warning(
                                f"Invalid cell value at ({r},{c}) for CF {cf_panel.get('id', i)} or TFT {tft_panel.get('id', j)}.")
                            valid_cell_structure = False
                            break

                        if cf_cell == 0 and tft_cell == 0:  # 양품 조건
                            current_yield += 1
                    if not valid_cell_structure: break

            if valid_cell_structure:
                yield_matrix[i][j] = current_yield
                if num_cf + num_tft < 20:
                    logger.debug(
                        f"Yield for CF {cf_panel.get('id', i)} - TFT {tft_panel.get('id', j)}: {current_yield}")
    return yield_matrix


class LcdMatchingSolver(BaseOrtoolsLinearSolver):
    """
    LCD CF-TFT 패널 매칭 문제를 정의하고 해결하는 클래스.
    OR-Tools의 MIP 솔버(CBC)를 사용합니다.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 패널 데이터를 파싱합니다.
        """
        super().__init__(input_data, 'CBC')

        # 입력 데이터 파싱
        self.cf_panels = self.input_data.get('cf_panels', [])
        self.tft_panels = self.input_data.get('tft_panels', [])
        self.num_cf = len(self.cf_panels)
        self.num_tft = len(self.tft_panels)

        # 모델링에 필요한 중간 데이터
        self.yield_matrix = []
        self.x = {}  # 결정 변수

    def _calculate_yield_matrix(self):
        """두 패널 타입 간의 수율(yield) 매트릭스를 계산합니다."""
        logger.solve("--- Pre-calculation: Yield Matrix ---")
        # create_cost_matrix 함수가 수율 매트릭스를 계산한다고 가정합니다.
        self.yield_matrix = create_cost_matrix(self.cf_panels, self.tft_panels)

    def _create_variables(self):
        """결정 변수 x_ij (CF_i와 TFT_j를 매칭하면 1)를 생성합니다."""
        logger.solve("--- 1. Creating Decision Variables ---")

        # 변수 생성 전, 수율 매트릭스 계산이 선행되어야 합니다.
        self._calculate_yield_matrix()

        for i in range(self.num_cf):
            for j in range(self.num_tft):
                # 수율이 0 이상인, 즉 유효한 매칭 쌍에 대해서만 변수를 생성합니다.
                if self.yield_matrix[i][j] >= 0:
                    self.x[i, j] = self.solver.BoolVar(f'x_{i}_{j}')

        if not self.x:
            raise ValueError("매칭 가능한 유효한 패널 쌍이 없습니다.")

        logger.debug(f"Created {len(self.x)} matching variables.")

    def _add_constraints(self):
        """각 패널은 최대 한 번만 매칭될 수 있다는 제약 조건을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")

        # 제약 1: 각 CF 패널은 최대 하나의 TFT 패널에만 할당됩니다.
        for i in range(self.num_cf):
            self.solver.Add(sum(self.x.get((i, j), 0) for j in range(self.num_tft)) <= 1)

        # 제약 2: 각 TFT 패널은 최대 하나의 CF 패널에만 할당됩니다.
        for j in range(self.num_tft):
            self.solver.Add(sum(self.x.get((i, j), 0) for i in range(self.num_cf)) <= 1)

        logger.debug(f"Added {self.num_cf + self.num_tft} matching constraints.")

    def _set_objective_function(self):
        """목표 함수 (총 수율 최대화)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")
        objective = self.solver.Objective()
        for (i, j), var in self.x.items():
            objective.SetCoefficient(var, float(self.yield_matrix[i][j]))
        objective.SetMaximization()
        logger.debug("Objective function set to maximize total yield.")

    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for LCD matching...")

        matched_pairs = []
        for (i, j), var in self.x.items():
            if var.solution_value() > 0.5:
                matched_pairs.append({
                    'cf': self.cf_panels[i],
                    'tft': self.tft_panels[j],
                    'cf_id': self.cf_panels[i].get('id', f'CF{i + 1}'),
                    'tft_id': self.tft_panels[j].get('id', f'TFT{j + 1}'),
                    'yield_value': self.yield_matrix[i][j]
                })

        return {
            'matched_pairs': matched_pairs,
            'total_yield': self.solver.Objective().Value()
        }
