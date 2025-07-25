from common_utils.common_data_utils import save_json_data
import logging
import random
import datetime

from core.decorators import log_data_creation

logger = logging.getLogger(__name__)


preset_trans_assign_items=3
preset_trans_assign_drivers=["김기사", "이배달", "박운송", "최신속", "정안전"]
preset_trans_assign_zones = ["강남구", "서초구", "송파구", "마포구", "영등포구"]
preset_trans_cost_matrix = [
    [47, 70, 30, 88, 25],
    [42, 58, 23, 91, 65],
    [89, 32, 92, 45, 55],
    [38, 66, 75, 29, 81],
    [51, 77, 48, 62, 39]
]

preset_num_resources=7
preset_num_projects=3
preset_resources = [
        {'id': 'R1', 'name': '김개발', 'cost': '100', 'skills': 'Python,ML'},
        {'id': 'R2', 'name': '이엔지', 'cost': '120', 'skills': 'Java,SQL,Cloud'},
        {'id': 'R3', 'name': '박기획', 'cost': '90', 'skills': 'SQL,Tableau'},
        {'id': 'R4', 'name': '최신입', 'cost': '70', 'skills': 'Python'},
        {'id': 'R5', 'name': '정고급', 'cost': '150', 'skills': 'Cloud,Python,K8s'},
        {'id': 'R6', 'name': '한디자', 'cost': '105', 'skills': 'UI,AWS,UX,React'},
        {'id': 'R7', 'name': '백엔드', 'cost': '110', 'skills': 'Java,Spring,SQL'},
        {'id': 'R8', 'name': '프론트', 'cost': '90', 'skills': 'React,JavaScript'},
        {'id': 'R9', 'name': '데브옵', 'cost': '140', 'skills': 'K8s,AWS,Cloud'},
        {'id': 'R10', 'name': '데이터', 'cost': '130', 'skills': 'SQL,Python,Tableau'},
    ]
preset_projects = [
        {'id': 'P1', 'name': 'AI 모델 개발', 'required_skills': 'Python,ML,SQL'},
        {'id': 'P2', 'name': '데이터베이스 마이그레이션', 'required_skills': 'AWS,SQL,Cloud'},
        {'id': 'P3', 'name': '웹 서비스 프론트엔드', 'required_skills': 'React,JavaScript'},
        {'id': 'P4', 'name': '클라우드 인프라 구축', 'required_skills': 'AWS,K8s'},
        {'id': 'P5', 'name': 'BI 대시보드 제작', 'required_skills': 'SQL,Tableau'},
    ]


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


@log_data_creation
def create_cf_tft_matching_json_data(source):
    num_cf = int(source.get('num_cf_panels', 3))
    num_tft = int(source.get('num_tft_panels', 3))
    rows = int(source.get('num_rows', 3))
    cols = int(source.get('num_cols', 3))
    rate = int(source.get('defect_rate', 10))

    generated_cf_panels = create_panel_data("CF", num_cf, rows, cols, rate)
    generated_tft_panels = create_panel_data("TFT", num_tft, rows, cols, rate)

    generated_data = {
        'problem_type': 'lcd_cf_tft',
        "num_cf_panels": num_cf,
        "num_tft_panels": num_tft,
        "num_rows": rows,
        "num_cols": cols,
        "defect_rate_percent": rate,
        "panel_dimensions": {"rows": rows, "cols": cols},
        "cf_panels": generated_cf_panels,
        "tft_panels": generated_tft_panels,
    }
    return generated_data


@log_data_creation
def create_transport_assignment_json_data(form_data):
    num_items = int(form_data.get('num_items'))

    drivers_data = []
    for i in range(num_items):
        drivers_data.append(form_data.get(f'driver_name_{i}'))

    zones_data = []
    for i in range(num_items):
        zones_data.append(form_data.get(f'zone_name_{i}'))

    cost_matrix = []
    for i in range(num_items):
        row = []
        for j in range(num_items):
            cost = int(form_data.get(f'cost_{i}_{j}'))
            row.append(cost)
        cost_matrix.append(row)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "driver_names": drivers_data,
        "zone_names": zones_data,
        "cost_matrix": cost_matrix,
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data


@log_data_creation
def create_resource_skill_matching_json_data(form_data):
    resources_data = []

    num_projects = int(form_data.get('num_projects'))
    selected_resource_ids = [sid for sid in form_data.getlist('selected_resources')]
    num_resources = len(selected_resource_ids)
    for i in range(num_resources):
        preset = preset_resources[i]
        resources_data.append({
            'id': preset['id'],
            'name': preset['name'],
            'cost': int(preset['cost']),
            'skills': [s.strip() for s in preset['skills'].split(',') if s.strip()]
        })

    projects_data = []
    for i in range(num_projects):
        req_skills_str = form_data.get(f'proj_{i}_required_skills', '')
        projects_data.append({
            'id': form_data.get(f'proj_{i}_id'),
            'name': form_data.get(f'proj_{i}_name'),
            'required_skills': [s.strip() for s in req_skills_str.split(',') if s.strip()]
        })

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_resources": num_resources,
        "num_projects": num_projects,
        "resources_data": resources_data,
        "projects_data": projects_data,
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }

    return input_data


def validate_required_skills(input_data):
    """
    각 프로젝트의 required_skills 중 resources_data의 skills에 없는 항목을 찾아 반환합니다.
    반환값: {스킬명: [포함하지 않은 프로젝트ID, ...], ...}
    """
    resources_data = input_data['resources_data']
    projects_data = input_data['projects_data']
    # 모든 리소스의 스킬을 집합으로 만듦
    all_skills = set()
    for res in resources_data:
        all_skills.update(res.get('skills', []))

    unmatched = {}
    for proj in projects_data:
        proj_id = proj.get('id')
        req_skills = set(proj.get('required_skills', []))
        missing = req_skills - all_skills
        for skill in missing:
            if skill not in unmatched:
                unmatched[skill] = []
            unmatched[skill].append(proj_id)

    # JSON을 key: value 형태의 HTML로 변환
    if isinstance(unmatched, dict):
        formatted_html = "<ul>"
        for k, v in unmatched.items():
            formatted_html += f"<li><strong>{k}</strong>: {v}</li>"
        formatted_html += "</ul>"
    elif isinstance(unmatched, list):
        formatted_html = "<ul>"
        for item in unmatched:
            if isinstance(item, dict):
                for k, v in item.items():
                    formatted_html += f"<li><strong>{k}</strong>: {v}</li>"
            else:
                formatted_html += f"<li>{item}</li>"
        formatted_html += "</ul>"
    else:
        formatted_html = str(unmatched)
    formatted_html = formatted_html.replace("'", "")
    return unmatched, formatted_html


def save_matching_assignment_json_data(input_data):
    problem_type = input_data.get('problem_type')
    model_data_type = f'matching_{problem_type}_data'
    filename_pattern = ''
    if "transport_assignment" == problem_type:
        num_driver = len(input_data.get('driver_names'))
        num_zone = len(input_data.get('zone_names'))
        filename_pattern = f"driver{num_driver}_zone{num_zone}"
    elif "resource_skill" == problem_type:
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        filename_pattern = f"resource{num_resources}_project{num_projects}"
    elif "lcd_cf_tft" == problem_type:
        num_cf_panels = input_data.get('num_cf_panels')
        num_tft_panels = input_data.get('num_tft_panels')
        num_rows = input_data.get('num_rows')
        num_cols = input_data.get('num_cols')
        defect_rate_percent = input_data.get('defect_rate_percent')
        filename_pattern = f"cf{num_cf_panels}_tft{num_tft_panels}_row{num_rows}_col{num_cols}_rate{str(defect_rate_percent).replace('.', 'p')}"
    return save_json_data(input_data, model_data_type, filename_pattern)

