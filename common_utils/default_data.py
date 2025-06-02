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

preset_depot_location = {"id": "D1","x": 79.0,"y": 73.0}
preset_customer_locations = [
        {"id": "C1","x": 103.0,"y": 120.0, "demand": 20},    {"id": "C2","x": 510.0,"y": 150.0, "demand": 20},
        {"id": "C3","x": 171.0,"y": 317.0, "demand": 20},    {"id": "C4","x": 486.0,"y": 283.0, "demand": 20}, 
        {"id": "C5","x": 384.0,"y": 45.0, "demand": 20},     {"id": "C6","x": 302.0,"y": 145.0, "demand": 20},
        {"id": "C7","x": 129.0,"y": 2211.0, "demand": 20},   {"id": "C8","x": 398.0,"y": 231.0, "demand": 20},
        {"id": "C9","x": 341.0,"y": 329.0, "demand": 20},    {"id": "C10","x": 537.0,"y": 365.0, "demand": 20}
    ]
preset_num_customers=5
preset_num_vehicles=3
preset_num_depots=1
preset_vehicle_capacity=100
