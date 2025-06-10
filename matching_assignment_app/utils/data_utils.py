from django.conf import settings

import os
import json
from common_utils.common_data_utils import save_json_data
import logging
import random
import datetime  # 파일명 생성 등에 사용 가능

logger = logging.getLogger(__name__)

def create_panel_data(panel_id_prefix, num_panels, rows, cols, rate):
    panels = []
    for i_panel in range(1, num_panels + 1):
        defect_map = []
        for r_idx in range(rows):
            row_map = []
            for c_idx in range(cols):
                if random.randint(1, 100) <= rate:
                    row_map.append(1)
                else:
                    row_map.append(0)
            defect_map.append(row_map)
        panels.append({
            "id": f"{panel_id_prefix}{i_panel}",
            "rows": rows,
            "cols": cols,
            "defect_map": defect_map
        })
    return panels


def create_matching_cf_tft_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate):
    generated_cf_panels = create_panel_data("CF", num_cf_panels, panel_rows, panel_cols, defect_rate)
    generated_tft_panels = create_panel_data("TFT", num_tft_panels, panel_rows, panel_cols, defect_rate)

    generated_data = {
        "panel_dimensions": {"rows": panel_rows, "cols": panel_cols},
        "cf_panels": generated_cf_panels,
        "tft_panels": generated_tft_panels,
        "settings": {
            "num_cf_panels": num_cf_panels,
            "num_tft_panels": num_tft_panels,
            "defect_rate_percent": defect_rate,
            "panel_rows": panel_rows,
            "panel_cols": panel_cols,
        }
    }
    return generated_data

def create_transport_assignment__json_data(form_data, submitted_num_items):
    num_items = submitted_num_items
    cost_matrix = [[0] * num_items for _ in range(num_items)]
    driver_names = []
    zone_names = []

    for i in range(num_items):
        driver_names.append(form_data.get(f'driver_name_{i}', f'기사 {i + 1}'))
        zone_names.append(form_data.get(f'zone_name_{i}', f'구역 {i + 1}'))
        for j in range(num_items):
            cost_val = form_data.get(f'cost_{i}_{j}')
            if cost_val is None or not cost_val.isdigit():
                raise ValueError(f"'{driver_names[i]}' -> '{zone_names[j]}' 비용이 유효한 숫자가 아닙니다.")
            cost_matrix[i][j] = int(cost_val)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "driver_names": driver_names,
        "zone_names": zone_names,
        "cost_matrix": cost_matrix,
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data

def save_matching_assignment_json_data(input_data):
    dir=''
    filename_pattern = ''
    if "TRAS" == input_data.get('problem_type'):
        num_driver = len(input_data.get('driver_names'))
        num_zone = len(input_data.get('zone_names'))
        dir = 'matching_transport_data'
        filename_pattern = f"driver{num_driver}_zone{num_zone}"


    return save_json_data(input_data, dir, filename_pattern)