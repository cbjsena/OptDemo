from django import template
import logging
logger = logging.getLogger(__name__)
register = template.Library()

@register.filter(name='get_range')
def get_range(number):
    logger.debug(f"get_range: number = {number} (type: {type(number)})")
    if isinstance(number, int) and number >= 0:
        logger.debug(f"get_range.isinstance: returning range({number})")
        return range(number)
    logger.warning(f"get_range: received invalid number '{number}', returning empty list.")
    return []

# get_cell_value는 이제 필터가 아니라 simple_tag로 만듭니다.
@register.simple_tag(name='get_cell_value_tag') # 태그 이름 변경 (선택 사항)
def get_cell_value_tag(matrix, row_idx, col_idx):
    logger.debug(
        f"get_cell_value_tag: matrix type = {type(matrix)}, matrix_preview = {str(matrix)[:50]}..., "
        f"row_idx = {row_idx} (type: {type(row_idx)}), "
        f"col_idx = {col_idx} (type: {type(col_idx)})"
    )
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
        logger.debug(f"Returning cell_value = {cell_value} (type: {type(cell_value)})")
        return cell_value
    except Exception as e:
        logger.error(f"Unexpected error in get_cell_value_tag for matrix, r:{row_idx}, c:{col_idx} - {e}", exc_info=True)
        return None