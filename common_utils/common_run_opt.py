import datetime
import logging
import os
from google.protobuf import text_format
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
logger = logging.getLogger(__name__)

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def start_log(problem_type:str):
    logger.info(f"Running Optimizer {problem_type}")


def solving_log(solver, problem_type:str, model=None):
    logger.info(f"Solving the {problem_type} model")
    if solver.__class__.__name__ == "CpSolver":
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
    else:
        status = solver.Solve()
        status_name = status_map.get(status, "UNKNOWN")
    processing_time = get_solving_time_sec(solver)
    logger.info(f"Solver finished. Status: {status_name}, Time: {processing_time} sec")

    return status, processing_time


def gurobi_solving_log(model, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    model.optimize()
    status = model.status
    processing_time = get_time(model.Runtime)
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return status, processing_time


def ortools_routing_solving_log(routing, search_parameters, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    status = routing.status()
    solve_end_time = datetime.datetime.now()
    processing_time = get_time((solve_end_time - solve_start_time).total_seconds())
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return solution, status, processing_time


def get_solving_time_sec(solver):
    """
       OR-Tools solver의 WallTime을 초 단위 문자열로 반환합니다.
       - CP-SAT solver는 초(sec)
       - Linear solver는 밀리초(ms)
       """
    try:
        time_raw = solver.WallTime()
    except AttributeError:
        return "N/A"

    # CP-SAT solver는 일반적으로 `CpSolver` 클래스의 인스턴스
    is_cp_sat = solver.__class__.__name__ == "CpSolver"

    processing_time = time_raw if is_cp_sat else time_raw / 1000  # 초 단위로 통일
    return get_time(processing_time)

def get_time(processing_time):
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"

def export_cp_model(model: cp_model.CpModel, filename: str):
    # 현재 파일 기준 상위 폴더의 mps 디렉토리 경로 구하기
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    mps_dir = os.path.abspath(os.path.join(base_dir, "..", "mps"))

    # 디렉토리가 없다면 생성
    os.makedirs(mps_dir, exist_ok=True)

    # 전체 경로 설정
    file_path = os.path.join(mps_dir, filename)

    proto = model.Proto()
    with open(file_path, "w") as f:
        f.write(text_format.MessageToString(proto))


def export_ortools_solver(solver: pywraplp.Solver, filename: str):
    # 현재 파일 기준 상위 폴더의 mps 디렉토리 경로 구하기
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    mps_dir = os.path.abspath(os.path.join(base_dir, "..", "mps"))

    # 디렉토리가 없다면 생성
    os.makedirs(mps_dir, exist_ok=True)

    # 전체 경로 설정
    file_path = os.path.join(mps_dir, filename)

    solver.WriteModelToMpsFile(file_path, True, False)