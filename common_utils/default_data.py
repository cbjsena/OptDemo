preset_budjet_items = [
            {'item_1_id': 'item_1', 'item_1_return_coeff': '3.1', 'item_1_min_alloc': '0', 'item_1_max_alloc': '200'},
            {'item_2_id': 'item_2', 'item_2_return_coeff': '2.1', 'item_2_min_alloc': '0', 'item_2_max_alloc': '300'},
            {'item_3_id': 'item_3', 'item_3_return_coeff': '1.1', 'item_3_min_alloc': '0', 'item_3_max_alloc': '1000'}
]

preset_datacenter_servers = [
            {'id': 'SrvA', 'cost': '500', 'cpu_cores': '48', 'ram_gb': '256', 'storage_tb': '10', 'power_kva': '0.5',
             'space_sqm': '0.2'},
            {'id': 'SrvB', 'cost': '300', 'cpu_cores': '32', 'ram_gb': '128', 'storage_tb': '5', 'power_kva': '0.3',
             'space_sqm': '0.1'},
            {'id': 'SrvC', 'cost': '800', 'cpu_cores': '128', 'ram_gb': '512', 'storage_tb': '20', 'power_kva': '0.8',
             'space_sqm': '0.3'}
]

preset_datacenter_services = [
            {'id': 'WebPool', 'revenue_per_unit': '100', 'req_cpu_cores': '4', 'req_ram_gb': '8',
             'req_storage_tb': '0.1', 'max_units': '50'},
            {'id': 'DBFarm', 'revenue_per_unit': '200', 'req_cpu_cores': '8', 'req_ram_gb': '16',
             'req_storage_tb': '0.5', 'max_units': '20'},
            {'id': 'BatchProc', 'revenue_per_unit': '150', 'req_cpu_cores': '16', 'req_ram_gb': '32',
             'req_storage_tb': '0.2', 'max_units': '30'}
]

preset_depot_location = {"id": "D1","x": 300.0,"y": 250.0}
preset_customer_locations = [
        {"id": "C1","x": 103.0,"y": 120.0, "demand": 30},    {"id": "C2","x": 510.0,"y": 150.0, "demand": 30},
        {"id": "C3","x": 171.0,"y": 317.0, "demand": 40},    {"id": "C4","x": 486.0,"y": 283.0, "demand": 40},
        {"id": "C5","x": 384.0,"y": 45.0, "demand": 30},     {"id": "C6","x": 302.0,"y": 145.0, "demand": 20},
        {"id": "C7","x": 129.0,"y": 221.0, "demand": 30},   {"id": "C8","x": 398.0,"y": 231.0, "demand": 30},
        {"id": "C9","x": 341.0,"y": 329.0, "demand": 20},    {"id": "C10","x": 537.0,"y": 365.0, "demand": 20}
    ]
preset_num_customers=5
preset_num_vehicles=3
preset_num_depots=1
preset_vehicle_capacity=100

preset_num_pairs=3
preset_pair_locations = [
        {"id": "Pair1","px": 103.0,"py": 120.0, "dx": 510.0,"dy": 150.0, "demand": 30},
        {"id": "Pair2","px": 171.0,"py": 317.0, "dx": 486.0,"dy": 283.0, "demand": 40},
        {"id": "Pair3","px": 384.0,"py": 45.0, "dx": 302.0,"dy": 145.0, "demand": 30},
        {"id": "Pair4","px": 129.0,"py": 221.0, "dx": 398.0,"dy": 231.0, "demand": 30},
        {"id": "Pair5","px": 341.0,"py": 329.0, "dx": 537.0,"dy": 365.0, "demand": 20},
    ]

preset_trans_assign_items=3
preset_trans_assign_drivers=["김기사", "이배달", "박운송", "최신속", "정안전"]
preset_trans_assign_zones = ["강남구", "서초구", "송파구", "마포구", "영등포구"]

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

preset_lot_sizing_num_periods = 6

preset_single_machine_num_jobs = 5
preset_single_machine_objective_choice = 'total_flow_time'
preset_single_machine_objective = [
            {'value': 'total_flow_time', 'name': '총 흐름 시간 최소화 (SPT)'},
            {'value': 'makespan', 'name': '총 완료 시간 최소화 (Makespan)'},
            {'value': 'total_tardiness', 'name': '총 지연 시간 최소화'}
        ]
preset_single_machine_data =[
    {'id': 'Job_1', 'processing_time': 5, 'due_date': 20},
    {'id': 'Job_2', 'processing_time': 12, 'due_date': 30},
    {'id': 'Job_3', 'processing_time': 8, 'due_date': 15},
    {'id': 'Job_4', 'processing_time': 3, 'due_date': 40},
    {'id': 'Job_5', 'processing_time': 15, 'due_date': 60},
    {'id': 'Job_6', 'processing_time': 6, 'due_date': 35},
    {'id': 'Job_7', 'processing_time': 9, 'due_date': 50},
    {'id': 'Job_8', 'processing_time': 4, 'due_date': 25},
    {'id': 'Job_9', 'processing_time': 11, 'due_date': 45},
    {'id': 'Job_10', 'processing_time': 7, 'due_date': 70},
]

preset_flow_shop_num_jobs = 4
preset_flow_shop_num_machines = 3
preset_flow_shop_data = [
    {'id': 'Job_1', 'processing_time': [29,78,9,36,49]},
    {'id': 'Job_2', 'processing_time': [43,92,8,45,68]},
    {'id': 'Job_3', 'processing_time': [90,85,87,32,91]},
    {'id': 'Job_4', 'processing_time': [77,39,55,64,82]},
    {'id': 'Job_5', 'processing_time': [95,13,47,84,22]},
    {'id': 'Job_6', 'processing_time': [20,88,70,69,74]},
    {'id': 'Job_7', 'processing_time': [58,48,85,6,86]},
    {'id': 'Job_8', 'processing_time': [73,10,29,76,4]},
    {'id': 'Job_9', 'processing_time': [36,2,31,75,59]},
    {'id': 'Job_10', 'processing_time': [12,88,58,99,9]}
]