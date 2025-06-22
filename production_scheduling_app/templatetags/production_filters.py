from django import template
import logging
logger = logging.getLogger(__name__)
register = template.Library()

@register.filter(name='get_range')
def get_range(number):
    # logger.debug(f"get_range: number = {number} (type: {type(number)})")
    if isinstance(number, int) and number >= 0:
        # logger.debug(f"get_range.isinstance: returning range({number})")
        return range(number)
    logger.warning(f"get_range: received invalid number '{number}', returning empty list.")
    return []

@register.filter
def get_item_simple(collection, key):
    # logger.debug(f"key: {key}, type:{type(key)}")
    if isinstance(collection, dict):
        return collection.get(key)
    if isinstance(collection, list):
        try:
            # key가 정수형 인덱스로 변환 가능한지 확인
            int_key = int(key)
            logger.debug(f"get_item_simple: Trying to access list with index {int_key}")

            # 인덱스가 유효한 범위 내에 있는지 확인
            if 0 <= int_key < len(collection):
                item = collection[int_key]
                logger.debug(f"get_item_simple: Success! Returning item: {item}")
                return item
            else:
                logger.warning(f"get_item_simple: Index {int_key} is out of bounds for list of size {len(collection)}.")
                return None

        except (ValueError, TypeError) as e:
            # key를 정수로 변환할 수 없거나 다른 타입 에러 발생 시
            logger.error(f"get_item_simple: Failed to convert key '{key}' to int or other type error. Error: {e}")
            return None
        except IndexError as e:
            # 인덱스 에러 발생 시 (위의 범위 체크로 대부분 방지되지만, 만약을 위해)
            logger.error(f"get_item_simple: IndexError for key '{key}'. Error: {e}")
            return None
    return None

# get_cell_value는 이제 필터가 아니라 simple_tag로 만듭니다.
@register.simple_tag(name='get_cell_value_tag') # 태그 이름 변경 (선택 사항)
def get_cell_value_tag(matrix, row_idx, col_idx):
    # logger.debug(
    #     f"get_cell_value_tag: matrix type = {type(matrix)}, matrix_preview = {str(matrix)[:50]}..., "
    #     f"row_idx = {row_idx} (type: {type(row_idx)}), "
    #     f"col_idx = {col_idx} (type: {type(col_idx)})"
    # )
    try:
        if not (isinstance(row_idx, int) and isinstance(col_idx, int)):
            logger.warning(f"Invalid index types: row_idx or col_idx is not an integer.")
            return None

        if matrix is None or not isinstance(matrix, list) or \
           row_idx < 0 or row_idx >= len(matrix) or \
           matrix[row_idx] is None or not isinstance(matrix[row_idx], list) or \
           col_idx < 0 or col_idx >= len(matrix[row_idx]):
            logger.warning(
                f"Index out of bounds or invalid matrix structure for get_cell_value_tag. "
                # 상세 로깅 추가
                f"Matrix len: {len(matrix) if isinstance(matrix, list) else 'N/A'}. "
                f"Requested row_idx: {row_idx}. "
                f"Row type: {type(matrix[row_idx]) if isinstance(matrix, list) and row_idx < len(matrix) else 'N/A'}. "
                f"Row len: {len(matrix[row_idx]) if isinstance(matrix, list) and row_idx < len(matrix) and isinstance(matrix[row_idx], list) else 'N/A'}. "
                f"Requested col_idx: {col_idx}."
            )
            return None

        cell_value = matrix[row_idx][col_idx]
        # logger.debug(f"Returning cell_value = {cell_value} (type: {type(cell_value)})")
        return cell_value
    except Exception as e:
        logger.error(f"Unexpected error in get_cell_value_tag for matrix, r:{row_idx}, c:{col_idx} - {e}", exc_info=True)
        return None

@register.simple_tag(name='make_key1')
def make_key1(prefix, index):
    return f"{prefix}_{index}"

@register.simple_tag(name='make_key2')
def make_key2(prefix, i, j):
    logger.info(f"---------make_key2: {prefix}_{i}_{j}")
    return f"{prefix}_{i}_{j}"

@register.filter(name='multiply_custom')
def multiply_custom(value, arg):
    """
    템플릿에서 두 값을 곱하는 커스텀 필터.
    예: {{ 3|multiply_custom:40 }} -> 120
    """
    try:
        # 입력값들을 숫자로 변환하여 곱셈 시도
        return float(value) * float(arg)
    except (ValueError, TypeError):
        # 변환 실패 시 0 또는 다른 기본값 반환
        return 0