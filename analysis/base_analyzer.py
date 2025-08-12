import logging
import uuid
from .models import Variable, Equation, MatrixEntry
from django.db import transaction


logger = logging.getLogger(__name__)

class BaseAnalyzer:
    """
    최적화 모델의 분석을 위한 기본 클래스.
    Gurobi 또는 OR-Tools 모델의 변수, 제약 조건 등을 데이터베이스에 기록합니다.
    공통 DB 연결 및 관리 기능을 제공하는 기본 클래스.
    db_config: dict with keys 'host', 'port', 'user', 'password', 'dbname'
    """
    def __init__(self, run_id):
        self.run_id = run_id if run_id else str(uuid.uuid4())
        logger.info(f"Initializing {self.__class__.__name__} with run_id: {self.run_id}")
        self._delete_existing()

    @transaction.atomic
    def _delete_existing(self):
        """
        분석 시작 전, 동일한 run_id의 이전 데이터를 모두 삭제
        """
        Variable.objects.filter(run_id=self.run_id).delete()
        Equation.objects.filter(run_id=self.run_id).delete()
        MatrixEntry.objects.filter(run_id=self.run_id).delete()


class BaseGurobiAnalyzer(BaseAnalyzer):
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        Gurobi 변수 객체(gp.Var)를 받아 DB에 저장합니다.
        :param var_obj: Gurobi 변수 객체
        :param var_group: 변수 그룹 이름
        :param kwargs: 추가 정보 (예: slot, home_team 등)
        :return: None
        """
        raise NotImplementedError

    @transaction.atomic
    def add_constraint(self,  model_obj, constr_obj, eq_group, **kwargs):
        """
        Gurobi 제약 객체(gp.Constr)를 받아 DB에 저장하고, matrix 항목도 함께 생성합니다.
        :param model_obj: Gurobi 모델 객체
        :param constr_obj: Gurobi 제약 객체
        :param eq_group: 제약 그룹 이름
        :param kwargs: 추가 정보 (예: slot, home_team 등)
        :return: None
        """
        raise NotImplementedError

    @transaction.atomic
    def update_variable_results(self, model_obj):
        """
        Gurobi 모델에서 변수 결과를 추출하여 DB에 저장합니다.
        :param model_obj: Gurobi 모델 객체
        :return: None
        """
        raise NotImplementedError


class BaseOrtoolsModelAnalyzer(BaseAnalyzer):
    """
    OR-Tools 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        OR-Tools 변수 객체를 받아 DB에 저장합니다.
        :param var_obj: OR-Tools CP-SAT 변수 객체
        :param var_group: 변수 그룹 이름
        :param kwargs: 추가 정보 (예: lower_bound, upper_bound 등)
        :return: None
        """
        raise NotImplementedError

    @transaction.atomic
    def add_constraint(self, eq_name, eq_group, **kwargs):
        """
        OR-Tools 정보를 받아 DB에 기록합니다.
        내부 구조 분석이 어려워 MatrixEntry는 생성하지 않습니다.
        :param eq_name: 제약 이름
        :param eq_group: 제약 그룹 이름
        :param kwargs: 추가 정보 (예: eq_type, sign, rhs 등)
        :return: None
        """
        raise NotImplementedError

    @transaction.atomic
    def update_variable_results(self, solver_obj, variables_to_log):
        """
        OR-Tools 모델에서 변수 결과를 추출하여 DB에 저장합니다.
        :param solver_obj: OR-Tools 모델 객체
        :param variables_to_log: 로그에 기록할 변수 목록
        :return: None
        """
        raise NotImplementedError


class BaseOrtoolsCpModelAnalyzer(BaseAnalyzer):
    """
    OR-Tools Cp 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        OR-Tools CP-SAT 변수 객체(cp_model.IntVar 또는 cp_model.BoolVar)를 받아 DB에 저장합니다.
        :param var_obj: OR-Tools CP-SAT 변수 객체
        :param var_group: 변수 그룹 이름
        :param kwargs: 추가 정보 (예: lower_bound, upper_bound 등)
        :return: None
        OR-Tools CP-SAT 변수 객체(cp_model.IntVar 또는 cp_model.BoolVar)
        """
        raise NotImplementedError

    @transaction.atomic
    def add_constraint(self, eq_name, eq_group, **kwargs):
        """
        OR-Tools 제약 객체(cp_model.Constraint)를 받아 DB에 기록합니다.
        내부 구조 분석이 어려워 MatrixEntry는 생성하지 않습니다.
        :param eq_name: 제약 이름
        :param eq_group: 제약 그룹 이름
        :param kwargs: 추가 정보 (예: eq_type, sign, rhs 등)
        :return: None
        """
        raise NotImplementedError

    @transaction.atomic
    def update_variable_results(self, solver_obj, variables_to_log):
        """
        OR-Tools CP-SAT 모델에서 변수 결과를 추출하여 DB에 저장합니다.
        :param solver_obj: OR-Tools CP-SAT solver 객체
        :param variables_to_log: 로그에 기록할 변수 목록
        :return: None
        """
        raise NotImplementedError