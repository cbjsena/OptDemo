import uuid
import logging
import uuid

from django.core.exceptions import ObjectDoesNotExist

from .models import Variable, Equation, MatrixEntry
from django.db import transaction
import gurobipy as gp

logger = logging.getLogger(__name__)


class SportsSchedulingGurobiAnalyzer:
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """

    def __init__(self, run_id=None):
        self.run_id = run_id if run_id else str(uuid.uuid4())
        logger.info(f"Initializing SportsSchedulingGurobiAnalyzer with run_id: {self.run_id}")
        self._delete_existing()

    def _delete_existing(self):
        # 분석 시작 전, 동일한 run_id의 이전 데이터를 모두 삭제
        Variable.objects.filter(run_id=self.run_id).delete()
        Equation.objects.filter(run_id=self.run_id).delete()
        MatrixEntry.objects.filter(run_id=self.run_id).delete()

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        Gurobi 변수 객체(gp.Var)를 받아 DB에 저장합니다.
        kwargs를 통해 slot, home_team 등 추가 정보를 받습니다.
        """
        var_type_map = {
            gp.GRB.CONTINUOUS: 'CONTINUOUS',
            gp.GRB.BINARY: 'BINARY',
            gp.GRB.INTEGER: 'INTEGER'
        }
        Variable.objects.create(
            run_id=self.run_id,
            var_name=var_obj.VarName,
            var_group=var_group,
            var_type=var_type_map.get(var_obj.VType, 'UNKNOWN'),
            lower_bound=var_obj.LB,
            upper_bound=var_obj.UB,
            # kwargs를 사용하여 동적으로 필드 값 할당
            slot=kwargs.get('slot'),
            home_team=kwargs.get('home_team'),
            away_team=kwargs.get('away_team'),
            team=kwargs.get('team'),
            city=kwargs.get('city'),
        )

    @transaction.atomic
    def add_constraint(self,  model_obj, constr_obj, eq_group, **kwargs):
        """Gurobi 제약 객체(gp.Constr)를 받아 DB에 저장하고, matrix 항목도 함께 생성합니다."""
        try:
            model_obj.update()
            eq_name = constr_obj.ConstrName
        except gp.GurobiError:
            logger.error("analyzer.add_constraint 에서 Gurobi 모델 업데이트 실패", exc_info=True)

        # 일반 제약(Linear Constraint) 처리
        if isinstance(constr_obj, gp.Constr):
            sign_map = {'<': '<=', '>': '>=', '=': '=='}
            sign = sign_map.get(constr_obj.Sense, '?')
            rhs = constr_obj.RHS
            eq_type = 'Linear'

            equation = Equation.objects.create(
                run_id=self.run_id, eq_name=eq_name, eq_group=eq_group,
                eq_type=eq_type, sign=sign, rhs=rhs,
                # [수정] kwargs를 사용하여 동적으로 필드 값 할당
                slot=kwargs.get('slot'),
                home_team=kwargs.get('home_team'),
                away_team=kwargs.get('away_team'),
                team=kwargs.get('team'),
                city=kwargs.get('city'),
            )

            row = model_obj.getRow(constr_obj)
            for i in range(row.size()):
                var = row.getVar(i)
                coeff = row.getCoeff(i)
                try:
                    variable = Variable.objects.get(run_id=self.run_id, var_name=var.VarName)
                    MatrixEntry.objects.create(
                        run_id=self.run_id,
                        variable=variable,
                        equation=equation,
                        coefficient=coeff
                    )
                except ObjectDoesNotExist:
                    logger.warning(f"Variable {var.VarName} not found for matrix entry")
        # 일반 제약(General Constraint) 처리 (예: AND, ABS 등)
        elif isinstance(constr_obj, gp.GenConstr):
            eq_type = f"General_{gp.GENCONSTR_NAMES.get(constr_obj.Type, 'UNKNOWN')}"
            Equation.objects.create(
                run_id=self.run_id, eq_name=eq_name, eq_group=eq_group,
                eq_type=eq_type, sign="", rhs=0.0,
                # [수정] kwargs를 사용하여 동적으로 필드 값 할당
                slot=kwargs.get('slot'),
                team=kwargs.get('team'),
            )

    @transaction.atomic
    def update_variable_results(self, model_obj):
        """최적화가 끝난 모델 객체를 받아 모든 변수의 결과 값을 DB에 업데이트합니다."""
        try:
            logger.solve(f"Updating results for run_id: {self.run_id}")

            # 1. 현재 run_id에 해당하는 모든 Variable 객체를 DB에서 가져옵니다.
            db_vars_queryset = Variable.objects.filter(run_id=self.run_id)

            # 2. var_name을 키로, Variable 객체를 값으로 하는 딕셔너리를 직접 만듭니다.
            all_db_vars = {var.var_name: var for var in db_vars_queryset}

            vars_to_update = []
            for var in model_obj.getVars():
                # 3. Gurobi 변수 이름으로 딕셔너리에서 해당 DB 객체를 찾습니다.
                if var.VarName in all_db_vars:
                    db_var = all_db_vars[var.VarName]
                    db_var.result_value = var.X
                    vars_to_update.append(db_var)

            if vars_to_update:
                Variable.objects.bulk_update(vars_to_update, ['result_value'])
            logger.info(f"Updated results for {len(vars_to_update)} variables.")
        except Exception as e:
            logger.error(f"update_variable_results: {e}")
            raise


class OrtoolsCpModelAnalyzer:
    """
    OR-Tools CP-SAT 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """

    def __init__(self, run_id=None):
        self.run_id = run_id if run_id else str(uuid.uuid4())
        logger.info(f"Initializing OrtoolsCpModelAnalyzer with run_id: {self.run_id}")
        Variable.objects.filter(run_id=self.run_id).delete()
        Equation.objects.filter(run_id=self.run_id).delete()
        MatrixEntry.objects.filter(run_id=self.run_id).delete()

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        OR-Tools 변수 객체를 받아 DB에 저장합니다.
        상한/하한 등은 kwargs를 통해 직접 전달받아야 합니다.
        """
        var_type = 'UNKNOWN'
        # 변수 타입 추정 (BoolVar, IntVar)
        if 'Bool' in str(type(var_obj)):
            var_type = 'BINARY'
        elif 'Int' in str(type(var_obj)):
            var_type = 'INTEGER'

        Variable.objects.create(
            run_id=self.run_id,
            var_name=var_obj.Name(),
            var_group=var_group,
            var_type=var_type,
            lower_bound=kwargs.get('lower_bound', 0),  # 호출 시 전달 필요
            upper_bound=kwargs.get('upper_bound', 1),  # 호출 시 전달 필요
            **kwargs
        )

    @transaction.atomic
    def add_constraint(self, constr_obj, eq_group, **kwargs):
        """
        OR-Tools 제약 객체를 받아 DB에 기록합니다.
        내부 구조 분석이 어려워 MatrixEntry는 생성하지 않습니다.
        """
        Equation.objects.create(
            run_id=self.run_id,
            eq_name=constr_obj.Name(),
            eq_group=eq_group,
            eq_type='CP_CONSTRAINT',  # CP-SAT 제약 유형
            sign=kwargs.get('sign', ''),
            rhs=kwargs.get('rhs', 0.0),
            **kwargs
        )

    @transaction.atomic
    def update_variable_results(self, solver_obj):
        """CP-SAT 솔버 객체를 받아 모든 변수의 결과 값을 DB에 업데이트합니다."""
        # CP-SAT는 모델이 아닌 솔버 객체에서 값을 읽습니다.
        vars_to_update = []
        # DB에 저장된 모든 변수를 가져옵니다.
        db_vars = Variable.objects.filter(run_id=self.run_id)

        for db_var in db_vars:
            # 솔버의 내부 프로토콜 버퍼에서 변수 이름으로 값을 찾습니다. (비효율적일 수 있음)
            # 이 부분은 실제 솔버의 변수 목록을 순회하는 것이 더 나을 수 있습니다.
            # 여기서는 개념적 구현을 위해 DB 기준으로 작성합니다.
            try:
                # 이 방식은 OR-Tools API에서 직접 지원하지 않으므로,
                # 실제로는 솔버 클래스에서 변수 객체와 결과값을 함께 전달해야 합니다.
                # 여기서는 개념만 보여줍니다.
                pass
            except Exception:
                pass
        # ... (실제 구현은 솔버 클래스와의 긴밀한 연동이 필요)
