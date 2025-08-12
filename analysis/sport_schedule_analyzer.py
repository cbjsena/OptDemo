import uuid
import logging
import uuid

from django.core.exceptions import ObjectDoesNotExist

from .base_analyzer import BaseGurobiAnalyzer, BaseOrtoolsCpModelAnalyzer
from .models import Variable, Equation, MatrixEntry
from django.db import transaction
import gurobipy as gp

logger = logging.getLogger(__name__)


class GurobiModelAnalyzer(BaseGurobiAnalyzer):
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        Gurobi 변수 객체(gp.Var)를 받아 DB에 저장합니다.
        kwargs를 통해 slot, home_team 등 추가 정보를 받습니다.
        """
        logger.solve(f"analysis_mode:add_variable-{var_group}")
        try:
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
                slot=kwargs.get('slot'),
                home_team=kwargs.get('home_team'),
                away_team=kwargs.get('away_team'),
                team=kwargs.get('team'),
                city=kwargs.get('city'),
            )
        except Exception as e:
            logger.error(f"Error adding variable {var_obj.Name()}: {e}")
            raise

    @transaction.atomic
    def add_constraint(self,  model_obj, constr_obj, eq_group, **kwargs):
        """Gurobi 제약 객체(gp.Constr)를 받아 DB에 저장하고, matrix 항목도 함께 생성합니다."""
        logger.solve(f"analysis_mode:add_constraint-{eq_group}")
        eq_name = ''
        try:
            model_obj.update()
            eq_name = constr_obj.ConstrName

            # 일반 제약(Linear Constraint) 처리
            if isinstance(constr_obj, gp.Constr):
                sign_map = {'<': '<=', '>': '>=', '=': '=='}
                sign = sign_map.get(constr_obj.Sense, '?')
                rhs = constr_obj.RHS
                eq_type = 'Linear'

                equation = Equation.objects.create(
                    run_id=self.run_id, eq_name=eq_name, eq_group=eq_group,
                    eq_type=eq_type, sign=sign, rhs=rhs,
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
        except gp.GurobiError as e:
            logger.error(f"Error adding constraint {eq_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding constraint {eq_name}: {e}")
            raise

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


class OrtoolsCpModelAnalyzer(BaseOrtoolsCpModelAnalyzer):
    """
    OR-Tools CP-SAT 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    @transaction.atomic
    def add_variable(self, var_obj, var_group, **kwargs):
        """
        OR-Tools 변수 객체를 받아 DB에 저장합니다.
        상한/하한 등은 kwargs를 통해 직접 전달받아야 합니다.
        """
        var_type = 'UNKNOWN'
        # 변수 타입 추정 (BoolVar, IntVar)
        try:
            if 'Bool' in str(type(var_obj)):
                var_type = 'BINARY'
            elif 'Int' in str(type(var_obj)):
                var_type = 'INTEGER'

            Variable.objects.create(
                run_id=self.run_id,
                var_name=var_obj.Name(),
                var_group=var_group,
                var_type=var_type,
                lower_bound=kwargs.get('lower_bound', 0),
                upper_bound=kwargs.get('upper_bound', 1),
                slot=kwargs.get('slot'),
                home_team=kwargs.get('home_team'),
                away_team=kwargs.get('away_team'),
                team=kwargs.get('team'),
                city=kwargs.get('city'),
            )
        except Exception as e:
            logger.error(f"Error adding variable {var_obj.Name()}: {e}")
            raise

    @transaction.atomic
    def add_constraint(self, eq_name, eq_group, **kwargs):
        """
        OR-Tools 제약 객체를 받아 DB에 기록합니다.
        내부 구조 분석이 어려워 MatrixEntry는 생성하지 않습니다.
        """
        try:
            Equation.objects.create(
                run_id=self.run_id,
                eq_name=eq_name,
                eq_group=eq_group,
                eq_type=kwargs.get('eq_type', ''),
                sign=kwargs.get('sign', ''),
                rhs=kwargs.get('rhs', 0.0),
                slot=kwargs.get('slot'),
                home_team=kwargs.get('home_team'),
                away_team=kwargs.get('away_team'),
                team=kwargs.get('team'),
                city=kwargs.get('city'),
            )
        except Exception as e:
            logger.error(f"Error adding constraint {eq_name}: {e}")
            raise

    @transaction.atomic
    def update_variable_results(self, solver_obj, variables_to_log):
        """CP-SAT 솔버 객체를 받아 모든 변수의 결과 값을 DB에 업데이트합니다."""
        try:
            logger.solve(f"Updating results for run_id: {self.run_id}")
            if not variables_to_log:
                logger.warning("No variables to log results for.")
                return
            # 1. 현재 run_id에 해당하는 모든 Variable 객체를 DB에서 가져옵니다.
            db_vars_queryset = Variable.objects.filter(run_id=self.run_id)

            # 2. var_name을 키로, Variable 객체를 값으로 하는 딕셔너리를 직접 만듭니다.
            all_db_vars = {var.var_name: var for var in db_vars_queryset}

            vars_to_update = []
            for var in variables_to_log:
                if var.Name() in all_db_vars:
                    db_var = all_db_vars[var.Name()]
                    db_var.result_value = solver_obj.Value(var)
                    vars_to_update.append(db_var)

            if vars_to_update:
                Variable.objects.bulk_update(vars_to_update, ['result_value'])
            logger.info(f"Updated results for {len(vars_to_update)} variables.")
        except Exception as e:
            logger.error(f"update_variable_results: {e}")
            raise
