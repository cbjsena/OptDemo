from ortools.sat.python import cp_model
from common_utils.common_run_opt import *
import logging

logger = logging.getLogger('resource_allocation_app')


class NurseRosteringSolver:
    """
       간호사 스케줄링 문제를 정의하고 해결하는 클래스.
       관련된 모든 데이터와 제약 조건 설정 함수들을 포함합니다.
       """

    def __init__(self, input_data):
        """
        생성자에서 입력 데이터를 파싱하고 모든 변수를 인스턴스 변수로 초기화합니다.
        """
        logger.info("Initializing Nurse Rostering Solver...")
        # --- 입력 데이터 파싱 및 인스턴스 변수 설정 ---
        self.problem_type = input_data['problem_type']
        self.nurses_data = input_data['nurses_data']
        self.num_nurses = input_data['num_nurses']
        self.num_days = input_data['num_days']
        self.shifts = input_data['shifts']
        self.skill_requirements = input_data['skill_requirements']
        self.vacation_requests = input_data['vacation_requests']
        self.enabled_fairness = input_data['enabled_fairness']
        self.weekend_days = input_data['weekend_days']
        self.min_shifts_per_nurse = input_data['min_shifts_per_nurse']
        self.max_shifts_per_nurse = input_data['max_shifts_per_nurse']

        # --- 파생 변수 설정 ---
        self.SHIFT_NIGHT = self.shifts[2]  # 예시: 야간 근무가 3번째 시프트라고 가정
        self.all_skills = list(self.skill_requirements[self.shifts[0]].keys())
        self.nurse_ids = [n['id'] for n in self.nurses_data]
        self.nurses_by_skill = {skill: [n['id'] for n in self.nurses_data if n['skill'] == skill] for skill in
                                self.all_skills}

        # --- 고정 값 또는 계산된 값 ---
        self.all_nurses = range(self.num_nurses)
        self.all_days = range(self.num_days)
        self.all_shifts = range(len(self.shifts))

        self.model = cp_model.CpModel()
        self.assigns ={}
        logger.info(f"Num nurses: {self.num_nurses}, Num days: {self.num_days}, Shifts: {self.shifts}")

    def _set_variables_assign(self):
        """
        특정 간호사를 특정 날짜, 특정 시프트에 배정하면 1, 아니면 0인 이진변수
        """
        logger.solve("--- Setting Variables assign ---")
        for n_id in self.nurse_ids:
            for d in self.all_days:
                for s in self.all_shifts:
                    varName = f"assigns_{self.nurses_data[n_id].get('name')}_{d + 1}_{self.shifts[s]}"
                    # logger.solve(f'BoolVar: {varName}')
                    self.assigns[(n_id, d, s)] = self.model.NewBoolVar(varName)

    def _set_constraints_day_work_one(self):
        """
        제약 1: 각 간호사는 하루 최대 1개 시프트 근무
        Hard constraint
        """
        logger.solve("--- Setting Equations DayWorkOne ---")
        for n_id in self.nurse_ids:
            for d in self.all_days:
                self.model.AddAtMostOne(self.assigns[(n_id, d, s)] for s in self.all_shifts)

    def _set_constraints_skill_req(self):
        """
        제약 2: 숙련도별 필요 인원 충족
        """
        logger.solve("--- Setting Equations SkillReq ---")
        for d in self.all_days:
            for s_idx, s_name in enumerate(self.shifts):
                for skill, required_count in self.skill_requirements[s_name].items():
                    nurses_with_that_skill = self.nurses_by_skill[skill]
                    self.model.Add(sum(self.assigns[(n_id, d, s_idx)] for n_id in nurses_with_that_skill) >= required_count)

    def _set_constraints_vacation_req(self):
        """
        제약 3: 휴가 요청 반영
        Hard constraint
        """
        logger.solve("--- Setting Equations Vacation ---")
        for n_id, off_days in self.vacation_requests.items():
            for d in off_days:
                self.model.Add(sum(self.assigns[(n_id, d, s)] for s in self.all_shifts) == 0)

    def _set_constraints_min_max_day_req(self):
        """
        제약 4: 간호사별 최소/최대 근무일 제약
        Hard constraint
        """
        logger.solve("--- Setting Equations Min Max Work Day ---")
        for n_id in self.nurse_ids:
            total_shifts_for_nurse = sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts)
            self.model.AddLinearConstraint(total_shifts_for_nurse, self.min_shifts_per_nurse, self.max_shifts_per_nurse)

    def _set_constrains_fair_nights(self):
        """
        목표 1: 공평한 야간 근무 분배 페널티 측정
        """
        logger.solve("--- Setting Fair Nights ---")
        if 'fair_nights' in self.enabled_fairness:
            night_shift_idx = self.shifts.index(self.SHIFT_NIGHT)
            night_shifts_worked = [sum(self.assigns[(n_id, d, night_shift_idx)] for d in self.all_days) for n_id in
                                   self.nurse_ids]
            min_nights = self.model.NewIntVar(0, self.num_days, 'min_nights')
            max_nights = self.model.NewIntVar(0, self.num_days, 'max_nights')
            self.model.AddMinEquality(min_nights, night_shifts_worked)
            self.model.AddMaxEquality(max_nights, night_shifts_worked)
            night_gap = max_nights - min_nights
        else:
            night_gap = 0

        return night_gap

    def _set_constrains_fair_offs(self):
        """
        목표 2: 공평한 휴무일 분배 페널티 측정
        """
        logger.solve("--- Setting Fair Offs ---")
        if 'fair_offs' in self.enabled_fairness:
            total_shifts_worked = [
                sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts) for n_id in
                self.nurse_ids]
            off_days_per_nurse = [self.num_days - s for s in total_shifts_worked]
            min_offs = self.model.NewIntVar(0, self.num_days, 'min_offs')
            max_offs = self.model.NewIntVar(0, self.num_days, 'max_offs')
            self.model.AddMinEquality(min_offs, off_days_per_nurse)
            self.model.AddMaxEquality(max_offs, off_days_per_nurse)
            off_gap = max_offs - min_offs
        else:
            off_gap = 0

        return off_gap

    def _set_constraints_fair_weekends(self):
        """
        목표 3: 공평한 주말 근무 분배
        """
        logger.solve("--- Setting Fair Weekends ---")
        if 'fair_weekends' in self.enabled_fairness:
            weekend_shifts_worked = [sum(self.assigns[(n_id, d, s)] for d in self.weekend_days for s in self.all_shifts )
                                     for n_id in self.nurse_ids]
            min_weekend_shifts = self.model.NewIntVar(0, len(self.weekend_days) * len(self.shifts), 'min_weekend')
            max_weekend_shifts = self.model.NewIntVar(0, len(self.weekend_days) * len(self.shifts), 'max_weekend')
            self.model.AddMinEquality(min_weekend_shifts, weekend_shifts_worked)
            self.model.AddMaxEquality(max_weekend_shifts, weekend_shifts_worked)
            weekend_gap = max_weekend_shifts - min_weekend_shifts
        else:
            weekend_gap = 0

        return weekend_gap

    def _set_constraints_over_shift(self):
        """
        목표 4: 초과 배정 최소화
        """
        logger.solve("--- Setting Over Shift ---")
        over_staffing_penalties = []
        for d in self.all_days:
            for s_idx, s_name in enumerate(self.shifts):
                # 해당 시프트에 배정된 총 인원
                total_assigned = sum(self.assigns[(n_id, d, s_idx)] for n_id in self.nurse_ids)
                # 해당 시프트의 최소 필요 총인원
                total_required = sum(self.skill_requirements[s_name].values())

                # 초과 인원에 대한 페널티 변수 생성
                over_staff = self.model.NewIntVar(0, self.num_nurses, f'over_staff_d{d}_s{s_idx}')
                self.model.Add(total_assigned - total_required <= over_staff)
                over_staffing_penalties.append(over_staff)
        return over_staffing_penalties

    def _set_objective_function(self):
        """목표 함수를 설정하고, 관련된 변수들을 반환합니다."""
        logger.solve("--- Setting Objective Function (Fairness) ---")

        # 야간 근무 공정성
        night_gap = 0
        if 'fair_nights' in self.enabled_fairness and self.SHIFT_NIGHT:
            night_gap = self._set_constrains_fair_nights()

        # 휴무일 공정성
        off_gap = 0
        if 'fair_offs' in self.enabled_fairness:
            off_gap = self._set_constrains_fair_offs()

        # 주말 근무 공정성
        weekend_gap = 0
        weekend_shifts_worked = None
        if 'fair_weekends' in self.enabled_fairness:
            weekend_gap = self._set_constraints_fair_weekends()

        over_staffing_penalties = self._set_constraints_over_shift()
        # 목표 함수 설정
        self.model.Minimize(night_gap * 2 + off_gap + weekend_gap * 3 + sum(over_staffing_penalties)*10)

    def solve(self):
        try:
            """
            전체 스케줄링 문제를 해결하는 메인 메서드.
            """
            self._set_variables_assign()
            self._set_constraints_day_work_one()
            self._set_constraints_skill_req()
            self._set_constraints_vacation_req()
            self._set_constraints_min_max_day_req()
            self._set_objective_function()
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 30.0
            status, processing_time = solving_log(solver, self.problem_type, self.model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                schedule = {}
                for d in self.all_days:
                    schedule[d] = {}
                    for s_idx, s_name in enumerate(self.shifts):
                        schedule[d][s_idx] = [self.nurses_data[n_id].get('name') for n_id in self.nurse_ids if solver.Value(self.assigns[(n_id, d, s_idx)]) == 1]

                # 각 간호사별 총 근무일 수
                total_shifts = [sum(
                    solver.Value(self.assigns[(n_id, d, s)]) for d in self.all_days for s in self.all_shifts)
                                for n_id in self.nurse_ids]

                # 각 간호사별 총 야간 근무일 수
                if self.SHIFT_NIGHT in self.shifts:
                    night_shift_idx = self.shifts.index(self.SHIFT_NIGHT)
                    total_nights = [
                        sum(solver.Value(self.assigns[(n_id, d, night_shift_idx)]) for d in self.all_days) for
                        n_id in self.nurse_ids]
                else:
                    total_nights = [0] * self.num_nurses

                # 각 간호사별 총 주말 근무일 수
                total_weekends = [sum(
                    solver.Value(self.assigns[(n_id, d, s)]) for d in self.weekend_days for s in
                    self.all_shifts) for n_id in self.nurse_ids]

                # 각 간호사별 총 휴무일 수
                total_offs = [self.num_days - ts for ts in total_shifts]

                results_data = {
                    'schedule': schedule,
                    'nurse_stats': {
                        n_id: {
                            'name': self.nurses_data[i]['name'],
                            'skill': self.nurses_data[i]['skill'],
                            'total': total_shifts[i],
                            'nights': total_nights[i],
                            'weekends': total_weekends[i],
                            'offs': total_offs[i]
                        } for i, n_id in enumerate(self.nurse_ids)
                    },
                    'total_penalty': solver.ObjectiveValue()
                }
                return results_data, None, processing_time
            else:
                return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", None
        except Exception as e:
            return None, f"오류 발생: {str(e)}", None
