import logging
from itertools import permutations

logger = logging.getLogger(__name__)


class PalletizingSolver:
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

        # 큰 상자를 먼저 배치해 적재율을 높입니다.
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
            },
            'placements': placed,
            'unplaced': unplaced,
        }

