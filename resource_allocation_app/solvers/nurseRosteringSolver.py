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
        self.num_nurses = input_data['num_nurses']
        self.num_days = input_data['num_days']
        self.shifts = input_data['shifts']
        self.weekend_days = input_data['weekend_days']
        self.min_shifts_per_nurse = input_data['min_shifts_per_nurse']
        self.max_shifts_per_nurse = input_data['max_shifts_per_nurse']

        self.nurses_data = input_data.get('nurses_data', [])
        self.skill_requirements = input_data.get('skill_requirements', {})
        self.vacation_requests = input_data.get('vacation_requests', {})
        self.enabled_fairness = input_data.get('enabled_fairness', [])
        self.day_shift_requests = input_data.get('day_shift_requests', [])

        # --- 파생 변수 설정 ---
        self.SHIFT_NIGHT = self.shifts[2]  # 예시: 야간 근무가 3번째 시프트라고 가정
        if len(self.skill_requirements) > 0:
            self.all_skills = list(self.skill_requirements[self.shifts[0]].keys())
            self.nurses_by_skill = {skill: [n['id'] for n in self.nurses_data if n['skill'] == skill] for skill in
                                    self.all_skills}

        # --- 고정 값 또는 계산된 값 ---
        self.nurse_ids = [n['id'] for n in self.nurses_data]
        self.all_nurses = range(self.num_nurses)
        self.all_days = range(self.num_days)
        self.all_shifts = range(len(self.shifts))
        self.num_shifts_per_day = len(self.shifts)

        self.model = cp_model.CpModel()
        self.assigns ={}
        logger.info(f"Num nurses: {self.num_nurses}, Num days: {self.num_days}, Shifts: {self.shifts}")

    def _set_variables_assign(self):
        """
        특정 간호사를 특정 날짜, 특정 시프트에 배정하면 1, 아니면 0인 이진변수
        """
        logger.solve("--- Setting Variables assign ---")
        try:
            for n_id in self.nurse_ids:
                for d in self.all_days:
                    for s in self.all_shifts:
                        varName = f"assigns_{n_id}_{d + 1}_{self.shifts[s]}"
                        # logger.solve(f'BoolVar: {varName}')
                        self.assigns[(n_id, d, s)] = self.model.NewBoolVar(varName)
        except Exception as e:
            logger.error(e)

    def _set_constraints_day_work_one(self):
        """
        제약 : 각 간호사는 하루 최대 1개 시프트 근무
        """
        logger.solve("--- Setting Equations DayWorkOne ---")
        try:
            for n_id in self.nurse_ids:
                for d in self.all_days:
                    self.model.AddAtMostOne(self.assigns[(n_id, d, s)] for s in self.all_shifts)
        except Exception as e:
            logger.error(e)

    def _set_constraints_skill_req(self):
        """
        제약 2: 숙련도별 필요 인원 충족
        """
        logger.solve("--- Setting Equations SkillReq ---")
        try:
            for d in self.all_days:
                for s_idx, s_name in enumerate(self.shifts):
                    for skill, required_count in self.skill_requirements[s_name].items():
                        nurses_with_that_skill = self.nurses_by_skill[skill]
                        self.model.Add(sum(self.assigns[(n_id, d, s_idx)] for n_id in nurses_with_that_skill) >= required_count)
        except Exception as e:
            logger.error(e)

    def _set_constraints_vacation_req(self):
        """
        제약 3: 휴가 요청 반영
        Hard constraint
        """
        logger.solve("--- Setting Equations Vacation ---")
        try:
            for n_id, off_days in self.vacation_requests.items():
                n_index = int(n_id)
                for d in off_days:
                    self.model.Add(sum(self.assigns[(n_index, d, s)] for s in self.all_shifts) == 0)
        except Exception as e:
            logger.error(e)

    def _set_constraints_no_night_followed_by_day(self):
        """
        제약: 야간 근무 다음날 근무 금지 (e.g. Ngt → Day/Aft 근무 방지)
        """
        logger.solve("--- Setting No Night Followed by Day ---")
        try:
            night_shift_idx = self.shifts.index(self.SHIFT_NIGHT)
            for n_id in self.nurse_ids:
                for d in range(self.num_days - 1):
                    worked_night = self.assigns[(n_id, d, night_shift_idx)]
                    worked_next_day = sum(self.assigns[(n_id, d + 1, s)] for s in self.all_shifts)
                    self.model.Add(worked_night + worked_next_day <= 1)
        except Exception as e:
            logger.error(f"[No Night Followed by Day] {e}")

    def _set_constraints_no_3_consecutive_shifts(self):
        """
        제약: 최소 휴식 보장 (근무 후 2개 시프트 휴식)
        """
        logger.solve("--- Setting No 3 Consecutive Shifts ---")
        try:
            for n_id in self.nurse_ids:
                for d in range(self.num_days - 1):
                    for s in self.all_shifts:
                        # 현재 시프트의 인덱스를 전체 시간 흐름으로 변환
                        current_shift_index = d * self.num_shifts_per_day + s

                        # 다음 2개 시프트에 대한 근무 금지 제약 추가
                        for i in range(1, 3):
                            next_shift_index = current_shift_index + i
                            next_d = next_shift_index // self.num_shifts_per_day
                            next_s = next_shift_index % self.num_shifts_per_day

                            if next_d < self.num_days:
                                # "간호사 n이 (d,s)에 근무하면, (next_d, next_s)에는 근무할 수 없다"
                                self.model.AddImplication(self.assigns[(n_id, d, s)], self.assigns[(n_id, next_d, next_s)].Not())
        except Exception as e:
            logger.error(f"[No 3-Day Consecutive Shifts] {e}")

    def _set_constraints_no_3_consecutive_days(self):
        """
        제약: 간호사가 3일 연속 근무하지 못하도록 금지
        """
        logger.solve("--- Setting No 3 Consecutive Days ---")
        try:
            for n_id in self.nurse_ids:
                for d in range(self.num_days - 2):
                    three_day_sum = sum(self.assigns[(n_id, d + offset, s)]
                                        for offset in range(3)
                                        for s in self.all_shifts)
                    self.model.Add(three_day_sum <= 2)
        except Exception as e:
            logger.error(f"[No 3-Day Consecutive Constraint] {e}")

    def _set_constraints_min_max_day_req(self):
        """
        제약 4: 간호사별 최소/최대 근무일 제약
        Hard constraint
        """
        logger.solve("--- Setting Equations Min Max Work Day ---")
        try:
            for n_id in self.nurse_ids:
                total_shifts_for_nurse = sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts)
                self.model.AddLinearConstraint(total_shifts_for_nurse, self.min_shifts_per_nurse, self.max_shifts_per_nurse)
        except Exception as e:
            logger.error(e)

    def _set_constrains_fair_nights(self):
        """
        목표 1: 공평한 야간 근무 분배 페널티 측정
        """
        logger.solve("--- Setting Fair Nights ---")
        try:
            night_shift_idx = self.shifts.index(self.SHIFT_NIGHT)
            # night_shifts_worked = 간호사 n의 야간 교대 근무 수 Y_n, 밸런스 변수
            night_shifts_worked = [sum(self.assigns[(n_id, d, night_shift_idx)] for d in self.all_days) for n_id in
                                   self.nurse_ids]
            min_nights = self.model.NewIntVar(0, self.num_days, 'min_nights')
            max_nights = self.model.NewIntVar(0, self.num_days, 'max_nights')
            self.model.AddMinEquality(min_nights, night_shifts_worked)  # min_nights <= Y_n, forall n in N
            self.model.AddMaxEquality(max_nights, night_shifts_worked)  # max_nights >= Y_n, forall n in N
            night_gap = max_nights - min_nights
        except Exception as e:
            logger.error(e)

        return night_gap

    def _set_constrains_fair_offs(self):
        """
        목표 2: 공평한 휴무일 분배 페널티 측정
        """
        logger.solve("--- Setting Fair Offs ---")
        try:
            # 별도 클래스와 장고에서 offset이 -로 다른 현상 발생하여 수정
            # 리스트 컴프리헨션 안에서 표현식이 평가되는 시점이 실행 환경에 따라 미세하게 다를 수 있습니다.
            # 특히 Jinja2 렌더링 또는 Django 초기화 과정에서 CP-SAT 객체가 미묘하게 달라지는 경우 발생
            # total_shifts_worked = [
            #     sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts)
            #     for n_id in self.nurse_ids]
            # 의도치 않게 식이 반대로 구성될 수 있음 (e.g. s - self.num_days)
            # off_days_per_nurse = [self.num_days - s for s in total_shifts_worked]

            off_days_per_nurse = [
                self.model.NewIntVar(0, self.num_days, f'off_days_{n_id}')
                for n_id in self.nurse_ids
            ]
            for i, n_id in enumerate(self.nurse_ids):
                self.model.Add(off_days_per_nurse[i] == self.num_days -
                               sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts))
            min_offs = self.model.NewIntVar(0, self.num_days, 'min_offs')
            max_offs = self.model.NewIntVar(0, self.num_days, 'max_offs')
            self.model.AddMinEquality(min_offs, off_days_per_nurse)
            self.model.AddMaxEquality(max_offs, off_days_per_nurse)
            off_gap = max_offs - min_offs
        except Exception as e:
            logger.error(e)

        return off_gap

    def _set_constraints_fair_weekends(self):
        """
        목표 3: 공평한 주말 근무 분배
        """
        logger.solve("--- Setting Fair Weekends ---")
        try:
            weekend_shifts_worked = [sum(self.assigns[(n_id, d, s)] for d in self.weekend_days for s in self.all_shifts )
                                     for n_id in self.nurse_ids]
            min_weekend_shifts = self.model.NewIntVar(0, len(self.weekend_days) * len(self.shifts), 'min_weekend_shifts')
            max_weekend_shifts = self.model.NewIntVar(0, len(self.weekend_days) * len(self.shifts), 'max_weekend_shifts')
            self.model.AddMinEquality(min_weekend_shifts, weekend_shifts_worked)
            self.model.AddMaxEquality(max_weekend_shifts, weekend_shifts_worked)
            weekend_gap = max_weekend_shifts - min_weekend_shifts
        except Exception as e:
            logger.error(e)

        return weekend_gap

    def _set_constraints_over_shift(self):
        """
        목표 4: 초과 배정 최소화
        """
        logger.solve("--- Setting Over Shift ---")
        try:
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
        except Exception as e:
            logger.error(e)
        return over_staffing_penalties

    def _set_constraints_req_shift(self):
        """
        제약 : 각 날짜, 각 시프트의 필요 인원 충족
        """
        logger.solve("--- Setting Equations ReqShift ---")
        try:
            for d in self.all_days:
                for s in self.all_shifts:
                    required_nurses = self.day_shift_requests[d][s]
                    self.model.Add(sum(self.assigns[(n_id, d, s)] for n_id in self.nurse_ids) == required_nurses)

        except Exception as e:
            logger.error(e)

    def _set_advanced_objective_function(self):
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

    def _set_advanced_result(self, solver):
        schedule = {}
        for d in self.all_days:
            schedule[d] = {}
            for s_idx, s_name in enumerate(self.shifts):
                schedule[d][s_idx] = [self.nurses_data[n_id].get('name') for n_id in self.nurse_ids if
                                      solver.Value(self.assigns[(n_id, d, s_idx)]) == 1]

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

        return results_data

    def _set_advanced_solve(self):
        try:
            """
            전체 스케줄링 문제를 해결하는 메인 메서드.
            """
            self._set_variables_assign()
            self._set_constraints_day_work_one()
            self._set_constraints_skill_req()
            self._set_constraints_vacation_req()
            self._set_constraints_min_max_day_req()
            self._set_constraints_no_night_followed_by_day()
            self._set_constraints_no_3_consecutive_days()
            self._set_advanced_objective_function()
            solver = cp_model.CpSolver()
            # export_model_proto(self.model, "local_model.pb.txt")
            # solver.parameters.log_search_progress = True  # 자세한 진행 출력
            solver.parameters.max_time_in_seconds = 30.0
            status, processing_time = solving_log(solver, self.problem_type, self.model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                results_data = self._set_advanced_result(solver)
                return results_data, None, processing_time
            else:
                return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", None
        except Exception as e:
            return None, f"오류 발생: {str(e)}", None

    def _set_objective_function(self):
        """목표 함수를 설정하고, 관련된 변수들을 반환합니다."""
        logger.solve("--- Setting Objective Function (Fairness) ---")

        # 목표 1: 근무일 수가 min/max 범위에서 벗어나는 정도 최소화
        total_shifts_worked = [sum(self.assigns[(n_id, d, s)] for d in self.all_days for s in self.all_shifts)
                               for n_id in self.nurse_ids]
        penalties = []
        for n_id in self.nurse_ids:
            under_penalty = self.model.NewIntVar(0, self.num_days, f'under_penalty_{n_id}')
            over_penalty = self.model.NewIntVar(0, self.num_days, f'over_penalty_{n_id}')
            self.model.Add(self.min_shifts_per_nurse - total_shifts_worked[n_id] <= under_penalty)
            self.model.Add(total_shifts_worked[n_id] - self.max_shifts_per_nurse <= over_penalty)
            penalties.append(under_penalty)
            penalties.append(over_penalty)

        # 주말 근무 공평 분배
        weekend_gap = self._set_constraints_fair_weekends()

        # --- 4. 목표 함수 설정 ---
        # 페널티 총합 + 주말 근무 격차 최소화 (주말 근무 공정성에 더 높은 가중치 부여)
        self.model.Minimize(sum(penalties) + weekend_gap * 2)

    def _set_result(self, solver):
        schedule = {}
        for d in self.all_days:
            schedule[d] = {}
            for s in self.all_shifts:
                schedule[d][s] = [n for n in self.nurse_ids if solver.Value(self.assigns[(n, d, s)]) == 1]

        total_shifts = [sum(
            solver.Value(self.assigns[(n_id, d, s)]) for d in self.all_days for s in self.all_shifts)
            for n_id in self.nurse_ids]

        total_weekends = [sum(
            solver.Value(self.assigns[(n_id, d, s)]) for d in self.weekend_days for s in
            self.all_shifts) for n_id in self.nurse_ids]

        results_data = {
            'schedule': schedule,
            'nurse_stats': {
                n_id: {
                    'name': self.nurses_data[i]['name'],
                    'total': total_shifts[i],
                    'weekends': total_weekends[i]
                } for i, n_id in enumerate(self.nurse_ids)
            },
            'nurse_work_days': total_shifts,
            'weekend_work_days': total_weekends,
            'total_penalty': solver.ObjectiveValue()
        }

        return results_data

    def _solve(self):
        try:
            """
            전체 스케줄링 문제를 해결하는 메인 메서드.
            """
            self._set_variables_assign()
            self._set_constraints_day_work_one()
            self._set_constraints_req_shift()
            self._set_constraints_no_3_consecutive_shifts()
            self._set_objective_function()
            solver = cp_model.CpSolver()
            export_model_proto(self.model, "local_model.pb.txt")
            # var_names, constraints = parse_pb_file("local_model.pb.txt")
            # desc_model_by_line(3571, var_names, constraints)
            # solver.parameters.log_search_progress = True  # 자세한 진행 출력
            solver.parameters.max_time_in_seconds = 30.0
            status, processing_time = solving_log(solver, self.problem_type, self.model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                results_data = self._set_result(solver)
                return results_data, None, processing_time
            else:
                return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", None
        except Exception as e:
            return None, f"오류 발생: {str(e)}", None
