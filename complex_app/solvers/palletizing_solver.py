import logging
from itertools import permutations

from common_utils.ortools_solvers import BaseOrtoolsCpSolver

logger = logging.getLogger(__name__)

class PalletizingSolver(BaseOrtoolsCpSolver):
    """
    OR-Tools CP-SAT 기반 3D 팔렛타이징 최적화 솔버
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        self.pallet = input_data['pallet']
        self.box_types = input_data['box_types']
        self.objective_function = input_data.get('objective_function', 'utilization')
        self.units = self._create_units()
        self.n = len(self.units)
        # 솔버 파라미터: 5% GAP 이내이면 조기 종료, 최대 60초
        self.relative_gap_limit = 0.05
        self.max_time_in_seconds = 60.0

    def _create_units(self):
        units = []
        for box_type in self.box_types:
            for idx in range(box_type['qty']):
                unit = {
                    'unit_id': f"{box_type['id']}-{idx + 1}",
                    'type_id': box_type['id'],
                    'l': int(box_type['l']),
                    'w': int(box_type['w']),
                    'h': int(box_type['h']),
                    'weight': int(box_type['weight']),
                    'rotatable': box_type['rotatable'],
                }
                units.append(unit)
        return units

    def _create_variables(self):
        pallet_l = int(self.pallet['l'])
        pallet_w = int(self.pallet['w'])
        pallet_h = int(self.pallet['h'])
        self.x = [self.model.NewBoolVar(f'x_{i}') for i in range(self.n)]
        self.pos_x = [self.model.NewIntVar(0, pallet_l, f'pos_x_{i}') for i in range(self.n)]
        self.pos_y = [self.model.NewIntVar(0, pallet_w, f'pos_y_{i}') for i in range(self.n)]
        self.pos_z = [self.model.NewIntVar(0, pallet_h, f'pos_z_{i}') for i in range(self.n)]
        # 회전 변수(단순화: rotatable이면 0/1, 아니면 0)
        self.rot = [self.model.NewBoolVar(f'rot_{i}') if self.units[i]['rotatable'] else None for i in range(self.n)]

        # 회전 고려한 유효 길이/폭 보조 변수
        self.eff_l = []
        self.eff_w = []
        for i in range(self.n):
            li = self.units[i]['l']
            wi = self.units[i]['w']
            if self.rot[i] is not None and li != wi:
                eff_l_i = self.model.NewIntVar(min(li, wi), max(li, wi), f'eff_l_{i}')
                eff_w_i = self.model.NewIntVar(min(li, wi), max(li, wi), f'eff_w_{i}')
                # rot=0 → eff_l=l, eff_w=w; rot=1 → eff_l=w, eff_w=l
                self.model.Add(eff_l_i == li).OnlyEnforceIf(self.rot[i].Not())
                self.model.Add(eff_l_i == wi).OnlyEnforceIf(self.rot[i])
                self.model.Add(eff_w_i == wi).OnlyEnforceIf(self.rot[i].Not())
                self.model.Add(eff_w_i == li).OnlyEnforceIf(self.rot[i])
                self.eff_l.append(eff_l_i)
                self.eff_w.append(eff_w_i)
            else:
                # 회전 불가 또는 l==w (회전해도 동일)
                self.eff_l.append(li)
                self.eff_w.append(wi)
                # 회전 불가면 rot 고정
                if self.rot[i] is not None and li == wi:
                    self.model.Add(self.rot[i] == 0)

    def _add_constraints(self):
        pallet_l = int(self.pallet['l'])
        pallet_w = int(self.pallet['w'])
        pallet_h = int(self.pallet['h'])
        max_weight = int(self.pallet['max_weight'])
        # 무게 제한
        self.model.Add(sum(self.units[i]['weight'] * self.x[i] for i in range(self.n)) <= max_weight)
        # 팔렛 내부에 위치 (eff_l, eff_w 사용)
        for i in range(self.n):
            h = self.units[i]['h']
            if isinstance(self.eff_l[i], int):
                self.model.Add(self.pos_x[i] + self.eff_l[i] <= pallet_l).OnlyEnforceIf(self.x[i])
                self.model.Add(self.pos_y[i] + self.eff_w[i] <= pallet_w).OnlyEnforceIf(self.x[i])
            else:
                self.model.Add(self.pos_x[i] + self.eff_l[i] <= pallet_l).OnlyEnforceIf(self.x[i])
                self.model.Add(self.pos_y[i] + self.eff_w[i] <= pallet_w).OnlyEnforceIf(self.x[i])
            self.model.Add(self.pos_z[i] + h <= pallet_h).OnlyEnforceIf(self.x[i])

        # 배치 안 된 박스는 위치 0으로 고정
        for i in range(self.n):
            self.model.Add(self.pos_x[i] == 0).OnlyEnforceIf(self.x[i].Not())
            self.model.Add(self.pos_y[i] == 0).OnlyEnforceIf(self.x[i].Not())
            self.model.Add(self.pos_z[i] == 0).OnlyEnforceIf(self.x[i].Not())

        # 겹침 방지
        for i in range(self.n):
            for j in range(i + 1, self.n):
                hi = self.units[i]['h']
                hj = self.units[j]['h']
                # 6방향 분리 조건
                no_overlap = [
                    self.model.NewBoolVar(f'no_{i}_{j}_0'),
                    self.model.NewBoolVar(f'no_{i}_{j}_1'),
                    self.model.NewBoolVar(f'no_{i}_{j}_2'),
                    self.model.NewBoolVar(f'no_{i}_{j}_3'),
                    self.model.NewBoolVar(f'no_{i}_{j}_4'),
                    self.model.NewBoolVar(f'no_{i}_{j}_5'),
                ]
                self.model.Add(self.pos_x[i] + self.eff_l[i] <= self.pos_x[j]).OnlyEnforceIf(no_overlap[0])
                self.model.Add(self.pos_x[j] + self.eff_l[j] <= self.pos_x[i]).OnlyEnforceIf(no_overlap[1])
                self.model.Add(self.pos_y[i] + self.eff_w[i] <= self.pos_y[j]).OnlyEnforceIf(no_overlap[2])
                self.model.Add(self.pos_y[j] + self.eff_w[j] <= self.pos_y[i]).OnlyEnforceIf(no_overlap[3])
                self.model.Add(self.pos_z[i] + hi <= self.pos_z[j]).OnlyEnforceIf(no_overlap[4])
                self.model.Add(self.pos_z[j] + hj <= self.pos_z[i]).OnlyEnforceIf(no_overlap[5])
                # 둘 다 배치된 경우 6개 중 하나는 참이어야 함
                self.model.AddBoolOr(no_overlap).OnlyEnforceIf([self.x[i], self.x[j]])

        # 지지 제약 (Support Constraint): 박스는 바닥(z=0) 또는 다른 박스 위에 놓여야 함
        # 아래 박스 윗면과 위 박스 바닥면의 겹침 면적이 위 박스 바닥 면적의 50% 이상이어야 함

        # 모든 박스의 end 좌표를 전역으로 미리 계산 (AddMinEquality 등에서 안전하게 사용)
        end_x = []
        end_y = []
        for i in range(self.n):
            ex = self.model.NewIntVar(0, pallet_l, f'end_x_{i}')
            ey = self.model.NewIntVar(0, pallet_w, f'end_y_{i}')
            self.model.Add(ex == self.pos_x[i] + self.eff_l[i])
            self.model.Add(ey == self.pos_y[i] + self.eff_w[i])
            end_x.append(ex)
            end_y.append(ey)

        for i in range(self.n):
            # on_floor ↔ pos_z[i] == 0 (양방향 채널링)
            on_floor = self.model.NewBoolVar(f'on_floor_{i}')
            self.model.Add(self.pos_z[i] == 0).OnlyEnforceIf(on_floor)
            self.model.Add(self.pos_z[i] >= 1).OnlyEnforceIf(on_floor.Not())

            li = self.units[i]['l']
            wi = self.units[i]['w']
            half_area_i = (li * wi) // 2  # 회전해도 면적 동일

            support_options = [on_floor]
            for j in range(self.n):
                if i == j:
                    continue
                hj = self.units[j]['h']
                sup = self.model.NewBoolVar(f'sup_{i}_by_{j}')
                support_options.append(sup)

                # sup=True → j의 윗면 == i의 바닥, j 배치됨
                self.model.Add(self.pos_z[j] + hj == self.pos_z[i]).OnlyEnforceIf(sup)
                self.model.AddImplication(sup, self.x[j])

                # X축 겹침 길이 (전역 보조 변수, sup 무관하게 안전)
                min_end_x = self.model.NewIntVar(0, pallet_l, f'mnex_{i}_{j}')
                self.model.AddMinEquality(min_end_x, [end_x[i], end_x[j]])
                max_start_x = self.model.NewIntVar(0, pallet_l, f'msx_{i}_{j}')
                self.model.AddMaxEquality(max_start_x, [self.pos_x[i], self.pos_x[j]])
                diff_x = self.model.NewIntVar(-pallet_l, pallet_l, f'dx_{i}_{j}')
                self.model.Add(diff_x == min_end_x - max_start_x)
                zero_var = self.model.NewConstant(0)
                overlap_x = self.model.NewIntVar(0, pallet_l, f'ox_{i}_{j}')
                self.model.AddMaxEquality(overlap_x, [diff_x, zero_var])

                # Y축 겹침 폭
                min_end_y = self.model.NewIntVar(0, pallet_w, f'mney_{i}_{j}')
                self.model.AddMinEquality(min_end_y, [end_y[i], end_y[j]])
                max_start_y = self.model.NewIntVar(0, pallet_w, f'msy_{i}_{j}')
                self.model.AddMaxEquality(max_start_y, [self.pos_y[i], self.pos_y[j]])
                diff_y = self.model.NewIntVar(-pallet_w, pallet_w, f'dy_{i}_{j}')
                self.model.Add(diff_y == min_end_y - max_start_y)
                overlap_y = self.model.NewIntVar(0, pallet_w, f'oy_{i}_{j}')
                self.model.AddMaxEquality(overlap_y, [diff_y, zero_var])

                # 겹침 면적 = overlap_x * overlap_y (전역)
                overlap_area = self.model.NewIntVar(0, pallet_l * pallet_w, f'oa_{i}_{j}')
                self.model.AddMultiplicationEquality(overlap_area, [overlap_x, overlap_y])

                # sup=True → 겹침 면적 >= 위 박스 바닥 면적의 50%
                self.model.Add(overlap_area >= half_area_i).OnlyEnforceIf(sup)

                # 모서리 정렬 제약: X축 또는 Y축의 모서리가 아래 박스와 같은 선상에 있어야 함
                # pos_x[i]==pos_x[j] OR end_x[i]==end_x[j] OR pos_y[i]==pos_y[j] OR end_y[i]==end_y[j]
                align = [
                    self.model.NewBoolVar(f'al_{i}_{j}_xl'),  # 왼쪽 모서리 정렬
                    self.model.NewBoolVar(f'al_{i}_{j}_xr'),  # 오른쪽 모서리 정렬
                    self.model.NewBoolVar(f'al_{i}_{j}_yf'),  # 앞 모서리 정렬
                    self.model.NewBoolVar(f'al_{i}_{j}_yb'),  # 뒤 모서리 정렬
                ]
                self.model.Add(self.pos_x[i] == self.pos_x[j]).OnlyEnforceIf(align[0])
                self.model.Add(end_x[i] == end_x[j]).OnlyEnforceIf(align[1])
                self.model.Add(self.pos_y[i] == self.pos_y[j]).OnlyEnforceIf(align[2])
                self.model.Add(end_y[i] == end_y[j]).OnlyEnforceIf(align[3])
                self.model.AddBoolOr(align).OnlyEnforceIf(sup)

            # 배치된 박스는 바닥이거나 다른 박스 위에 있어야 함
            self.model.AddBoolOr(support_options).OnlyEnforceIf(self.x[i])

    def _set_objective_function(self):
        # 목적함수: 적재 부피, 박스 수, 무게 등
        if self.objective_function == 'utilization':
            self.model.Maximize(sum(self.units[i]['l'] * self.units[i]['w'] * self.units[i]['h'] * self.x[i] for i in range(self.n)))
        elif self.objective_function == 'boxes':
            self.model.Maximize(sum(self.x[i] for i in range(self.n)))
        elif self.objective_function == 'weight':
            self.model.Maximize(sum(self.units[i]['weight'] * self.x[i] for i in range(self.n)))
        else:
            self.model.Maximize(sum(self.x[i] for i in range(self.n)))

    def _extract_results(self, solver):
        placed = []
        unplaced = []
        total_weight = 0
        pallet_l = int(self.pallet['l'])
        pallet_w = int(self.pallet['w'])
        pallet_h = int(self.pallet['h'])
        for i in range(self.n):
            if solver.Value(self.x[i]) == 1:
                # 유효 길이/폭 추출
                if isinstance(self.eff_l[i], int):
                    l = self.eff_l[i]
                    w = self.eff_w[i]
                else:
                    l = solver.Value(self.eff_l[i])
                    w = solver.Value(self.eff_w[i])
                placed.append({
                    'unit_id': self.units[i]['unit_id'],
                    'type_id': self.units[i]['type_id'],
                    'x': solver.Value(self.pos_x[i]),
                    'y': solver.Value(self.pos_y[i]),
                    'z': solver.Value(self.pos_z[i]),
                    'l': l,
                    'w': w,
                    'h': self.units[i]['h'],
                    'weight': self.units[i]['weight'],
                })
                total_weight += self.units[i]['weight']
            else:
                unplaced.append({
                    'unit_id': self.units[i]['unit_id'],
                    'reason': 'not_selected',
                })
        pallet_volume = pallet_l * pallet_w * pallet_h
        loaded_volume = sum(box['l'] * box['w'] * box['h'] for box in placed)
        utilization = (loaded_volume / pallet_volume * 100.0) if pallet_volume > 0 else 0.0
        max_height = max((box['z'] + box['h'] for box in placed), default=0)
        return {
            'summary': {
                'total_units': self.n,
                'placed_units': len(placed),
                'unplaced_units': len(unplaced),
                'loaded_volume': round(loaded_volume, 2),
                'pallet_volume': round(pallet_volume, 2),
                'utilization_percent': round(utilization, 2),
                'total_weight': round(total_weight, 2),
                'max_weight': int(self.pallet['max_weight']),
                'max_height': round(max_height, 2),
            },
            'placements': placed,
            'unplaced': unplaced,
        }

    # solve()는 BaseOrtoolsCpSolver의 기본 구현을 사용합니다.

class PalletizingLogicSolver:
    """간단한 3D 팔렛타이징(그리디 + extreme points) 데모 솔버."""

    def __init__(self, input_data):
        self.input_data = input_data
        self.pallet = input_data['pallet']
        self.box_types = input_data['box_types']

    @staticmethod
    def _overlap_1d(a_start, a_end, b_start, b_end):
        return a_start < b_end and b_start < a_end

    def _is_non_overlapping(self, candidate, placed):
        cx, cy, cz = candidate['x'], candidate['y'], candidate['z']
        cl, cw, ch = candidate['l'], candidate['w'], candidate['h']

        for box in placed:
            bx, by, bz = box['x'], box['y'], box['z']
            bl, bw, bh = box['l'], box['w'], box['h']
            overlap_x = self._overlap_1d(cx, cx + cl, bx, bx + bl)
            overlap_y = self._overlap_1d(cy, cy + cw, by, by + bw)
            overlap_z = self._overlap_1d(cz, cz + ch, bz, bz + bh)
            if overlap_x and overlap_y and overlap_z:
                return False
        return True

    def _orientations(self, box_type):
        dims = (box_type['l'], box_type['w'], box_type['h'])
        if box_type['rotatable']:
            return sorted(set(permutations(dims, 3)))
        return [dims]

    def solve(self):
        pallet_l = self.pallet['l']
        pallet_w = self.pallet['w']
        pallet_h = self.pallet['h']
        max_weight = self.pallet['max_weight']

        units = []
        for box_type in self.box_types:
            for idx in range(box_type['qty']):
                units.append({
                    'unit_id': f"{box_type['id']}-{idx + 1}",
                    'type_id': box_type['id'],
                    'l': box_type['l'],
                    'w': box_type['w'],
                    'h': box_type['h'],
                    'weight': box_type['weight'],
                    'rotatable': box_type['rotatable'],
                })

        # Advanced: 목적 함수에 따라 정렬
        if self.input_data.get('problem_type') == 'advanced_palletizing':
            obj = self.input_data.get('objective_function', 'utilization')
            if obj == 'utilization':
                units.sort(key=lambda x: x['l'] * x['w'] * x['h'], reverse=True)  # 큰 부피부터
            elif obj == 'boxes':
                units.sort(key=lambda x: x['l'] * x['w'] * x['h'])  # 작은 부피부터
            elif obj == 'weight':
                units.sort(key=lambda x: x['weight'], reverse=True)  # 무거운 것부터
        else:
            # 기존: 큰 상자부터
            units.sort(key=lambda x: x['l'] * x['w'] * x['h'], reverse=True)

        placed = []
        unplaced = []
        extreme_points = [(0, 0, 0)]
        total_weight = 0.0

        for unit in units:
            if total_weight + unit['weight'] > max_weight:
                unplaced.append({
                    'unit_id': unit['unit_id'],
                    'reason': 'weight_limit',
                })
                continue

            placed_this_unit = False
            for px, py, pz in sorted(set(extreme_points), key=lambda p: (p[2], p[1], p[0])):
                for ol, ow, oh in self._orientations(unit):
                    if px + ol > pallet_l or py + ow > pallet_w or pz + oh > pallet_h:
                        continue

                    candidate = {
                        'unit_id': unit['unit_id'],
                        'type_id': unit['type_id'],
                        'x': px,
                        'y': py,
                        'z': pz,
                        'l': ol,
                        'w': ow,
                        'h': oh,
                        'weight': unit['weight'],
                    }
                    if not self._is_non_overlapping(candidate, placed):
                        continue

                    placed.append(candidate)
                    total_weight += unit['weight']
                    placed_this_unit = True

                    # 새 extreme point 추가
                    extreme_points.extend([
                        (px + ol, py, pz),
                        (px, py + ow, pz),
                        (px, py, pz + oh),
                    ])
                    break

                if placed_this_unit:
                    break

            if not placed_this_unit:
                unplaced.append({
                    'unit_id': unit['unit_id'],
                    'reason': 'no_feasible_position',
                })

        pallet_volume = pallet_l * pallet_w * pallet_h
        loaded_volume = sum(box['l'] * box['w'] * box['h'] for box in placed)
        utilization = (loaded_volume / pallet_volume * 100.0) if pallet_volume > 0 else 0.0
        max_height = max((box['z'] + box['h'] for box in placed), default=0)

        return {
            'summary': {
                'total_units': len(units),
                'placed_units': len(placed),
                'unplaced_units': len(unplaced),
                'loaded_volume': round(loaded_volume, 2),
                'pallet_volume': round(pallet_volume, 2),
                'utilization_percent': round(utilization, 2),
                'total_weight': round(total_weight, 2),
                'max_weight': max_weight,
                'max_height': round(max_height, 2),
            },
            'placements': placed,
            'unplaced': unplaced,
        }
