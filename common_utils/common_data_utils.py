from django.conf import settings

import os
import json
import logging

logger = logging.getLogger(__name__)


def save_json_data(generated_data, model_data_type, filename_pattern):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    data_dir_path_str = settings.DEMO_DIR_MAP[model_data_type]
    if not data_dir_path_str:
        logger.warning(f"{data_dir_path_str} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)

        # 1. 먼저 저장할 고유한 파일 경로를 찾습니다.
        filepath = ''
        seq = 0
        while True:
            if seq == 0:
                potential_filename = f"{filename_pattern}.json"
            else:
                potential_filename = f"{filename_pattern}_seq{seq}.json"

            filepath = os.path.join(data_dir_path_str, potential_filename)
            if not os.path.exists(filepath):
                break  # 사용할 파일 경로를 찾았으므로 루프 종료
            seq += 1
            if seq > 100:  # 무한 루프 방지
                raise IOError("Could not find a unique filename after 100 attempts.")

        # 2. 찾은 경로에 파일을 저장합니다.
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Input data saved successfully to: {filepath}")
        return get_save_info(filepath), None
    except IOError as e:
        logger.error(f"Failed to save input data to {filepath}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"


def get_save_info(filepath):
    return f"입력 데이터가 '{filepath}'으로 서버에 저장되었습니다."