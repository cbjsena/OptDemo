import uuid
import logging
import uuid

from django.core.exceptions import ObjectDoesNotExist

from .models import Variable, Equation, MatrixEntry
from django.db import transaction
import gurobipy as gp

logger = logging.getLogger(__name__)

class GurobiModelAnalyzer:
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """

    def __init__(self, run_id=None):
        self.run_id = run_id if run_id else str(uuid.uuid4())
        logger.info(f"Initializing GurobiModelAnalyzer with run_id: {self.run_id}")
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
        vars_to_update = []
        all_db_vars = Variable.objects.filter(run_id=self.run_id).in_bulk()

        for var in model_obj.getVars():
            if var.VarName in all_db_vars:
                db_var = all_db_vars[var.VarName]
                db_var.result_value = var.X
                vars_to_update.append(db_var)

        if vars_to_update:
            Variable.objects.bulk_update(vars_to_update, ['result_value'])
        logger.info(f"Updated results for {len(vars_to_update)} variables.")
