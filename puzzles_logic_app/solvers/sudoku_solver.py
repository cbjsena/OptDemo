import logging
import math
import random

from ortools.sat.python import cp_model
from common_utils.ortools_solvers import BaseOrtoolsCpSolver

logger = logging.getLogger('puzzles_logic_app')


class SudokuSolver(BaseOrtoolsCpSolver):
    """
    모든 크기(NxN)의 스도쿠 퍼즐을 해결하는 클래스.
    N은 반드시 완전 제곱수여야 합니다 (예: 9, 16, 25).
    """

    def __init__(self, input_data):
        """
        생성자: BaseCpSolver를 초기화하고 스도쿠 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)
        self.input_grid = self.input_data.get('input_grid')
        self.num_size = self.input_data.get('num_size')
        self.subgrid_size = int(math.sqrt(self.num_size))

        # 모델링에 사용할 변수 초기화
        self.grid_vars = {}

    def _create_variables(self):
        """결정 변수 grid[(row, col)] (각 셀의 숫자)를 생성합니다."""
        logger.solve("--- 1. Creating Sudoku Variables ---")
        for r in range(self.num_size):
            for c in range(self.num_size):
                self.grid_vars[(r, c)] = self.model.NewIntVar(1, self.num_size, f'cell_{r}_{c}')

    def _add_constraints(self):
        """스도쿠 규칙(행, 열, 서브그리드)과 초기 숫자 제약을 추가합니다."""
        logger.solve("--- 2. Adding Sudoku Constraints ---")

        # 각 행(row)의 모든 숫자는 달라야 합니다.
        for r in range(self.num_size):
            self.model.AddAllDifferent([self.grid_vars[(r, c)] for c in range(self.num_size)])

        # 각 열(column)의 모든 숫자는 달라야 합니다.
        for c in range(self.num_size):
            self.model.AddAllDifferent([self.grid_vars[(r, c)] for r in range(self.num_size)])

        # 각 서브그리드의 모든 숫자는 달라야 합니다.
        for sub_r in range(0, self.num_size, self.subgrid_size):
            for sub_c in range(0, self.num_size, self.subgrid_size):
                subgrid = [
                    self.grid_vars[(r, c)]
                    for r in range(sub_r, sub_r + self.subgrid_size)
                    for c in range(sub_c, sub_c + self.subgrid_size)
                ]
                self.model.AddAllDifferent(subgrid)

        # 초기 퍼즐의 주어진 숫자들을 제약으로 추가합니다.
        for r in range(self.num_size):
            for c in range(self.num_size):
                if self.input_grid[r][c] != 0:
                    self.model.Add(self.grid_vars[(r, c)] == self.input_grid[r][c])

    def _set_objective_function(self):
        """
        스도쿠는 만족 문제(satisfaction problem)이므로, 별도의 목적 함수는 없습니다.
        """
        logger.solve("--- 3. No Objective Function for Sudoku (Satisfaction Problem) ---")
        pass

    def _extract_results(self, solver):
        """솔버 실행 후, 완성된 스도쿠 그리드를 추출합니다."""
        logger.info("Extracting solved Sudoku grid...")
        solved_grid = [
            [solver.Value(self.grid_vars[(r, c)]) for c in range(self.num_size)]
            for r in range(self.num_size)
        ]
        return solved_grid

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """솔버의 해를 세는 콜백 클래스"""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        # 2개 이상의 해를 찾으면 더 이상 탐색할 필요 없음
        if self.__solution_count >= 2:
            self.StopSearch()

        # 첫 번째 해는 저장해둠
        if self.__solution_count == 1:
            solution = {}
            for v in self.__variables:
                solution[v] = self.Value(v)
            self.solutions.append(solution)

    def solution_count(self):
        return self.__solution_count


def has_unique_solution(board):
    """주어진 스도쿠 퍼즐의 해가 유일한지 검사합니다."""
    N = len(board)
    model = cp_model.CpModel()
    grid = {}
    for row in range(N):
        for col in range(N):
            grid[(row, col)] = model.NewIntVar(1, N, f'grid_{row}_{col}')

    # 기존 제약 조건 추가
    for row in range(N):
        for col in range(N):
            if board[row][col] != 0:
                model.Add(grid[(row, col)] == board[row][col])
    for row in range(N):
        model.AddAllDifferent([grid[(row, col)] for col in range(N)])
        model.AddAllDifferent([grid[(col, row)] for col in range(N)])

    subgrid_size = int(N ** 0.5)
    for row_idx in range(0, N, subgrid_size):
        for col_idx in range(0, N, subgrid_size):
            subgrid_vars = [grid[(row_idx + row, col_idx + col)] for row in range(subgrid_size) for col in range(subgrid_size)]
            model.AddAllDifferent(subgrid_vars)

    solver = cp_model.CpSolver()
    # 해를 2개까지만 찾도록 설정
    solution_printer = VarArraySolutionPrinter(list(grid.values()))
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    processing_time = solver.WallTime()
    logger.debug(f"Solver status: {status}, Time: {processing_time} sec")

    return solution_printer.solution_count() == 1


def generate_sudoku(difficulty='medium', num_size=9):
    """
    주어진 난이도에 맞춰 새로운 스도쿠 퍼즐을 생성합니다.
    """
    # 1. 완전한 정답 그리드 생성
    # 빈 그리드에서 시작하여 OR-Tools가 하나의 해를 찾도록 함
    subgrid_size = int(math.sqrt(num_size))
    total_cell_size = num_size*num_size
    first_row = random.sample(range(1, 10), 9)
    randomized_start_board = [first_row] + [[0] * 9 for _ in range(8)]
    input_data = {
        'problem_type': 'sudoku',
        'input_grid': randomized_start_board,
        'difficulty': difficulty,
        'num_size': num_size,
        'subgrid_size': subgrid_size
    }
    # SudokuSolver 인스턴스를 생성한 후, .solve() 메서드를 호출합니다.
    solver_instance = SudokuSolver(input_data)
    solution_board, _, _ = solver_instance.solve()

    puzzle = [row[:] for row in solution_board]

    # 난이도별 제거할 셀의 개수 설정
    if difficulty == 'easy':
        # 36-45 clues
        cells_to_remove = total_cell_size - random.randint(int(total_cell_size*0.4), int(total_cell_size*0.5))
    elif difficulty == 'hard':
        # 22-27 clues
        cells_to_remove = total_cell_size - random.randint(int(total_cell_size*0.2), int(total_cell_size*0.3))
    else:  # medium
        # 28-35 clues
        cells_to_remove = total_cell_size - random.randint(int(total_cell_size*0.3), int(total_cell_size*0.4))

    # 2. 유일해를 유지하며 숫자 제거
    coords = [(r, c) for r in range(num_size) for c in range(num_size)]
    random.shuffle(coords)

    removed_count = 0
    for r, c in coords:
        if removed_count >= cells_to_remove:
            break

        original_value = puzzle[r][c]
        puzzle[r][c] = 0

        if not has_unique_solution(puzzle):
            # 해가 유일하지 않으면 숫자를 다시 복원
            puzzle[r][c] = original_value
        else:
            removed_count += 1

    return puzzle


def create_puzzle_from_solution(solution_grid, difficulty='medium'):
    """
    완성된 정답 그리드에서 난이도에 따라 무작위로 숫자를 제거하여
    새로운 퍼즐을 생성합니다.
    """
    if not solution_grid:
        return []

    N = len(solution_grid)
    puzzle = [row[:] for row in solution_grid]  # 정답 그리드 복사

    # 난이도에 따라 남길 숫자의 비율(%) 설정
    if difficulty == 'easy':
        reveal_ratio = 0.45
    elif difficulty == 'hard':
        reveal_ratio = 0.25
    else:  # medium
        reveal_ratio = 0.35

    total_cells = N * N
    cells_to_reveal = int(total_cells * reveal_ratio)

    all_coords = [(r, c) for r in range(N) for c in range(N)]
    random.shuffle(all_coords)

    coords_to_blank = all_coords[cells_to_reveal:]
    for r, c in coords_to_blank:
        puzzle[r][c] = 0

    return puzzle