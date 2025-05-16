from django import template

register = template.Library()

@register.filter(name='get_range')
def get_range(number):
    """
    템플릿에서 숫자를 받아 range(숫자)를 반환합니다.
    예: {% for i in 5|get_range %} -> i는 0, 1, 2, 3, 4
    """
    if isinstance(number, int):
        return range(number)
    return []

@register.filter(name='get_cell_value')
def get_cell_value(matrix, indices_str):# "row,col" 형태의 문자열 인자를 받음
    try:
        row_idx, col_idx = map(int, str(indices_str).split(',')) # str() 추가하여 안전성 확보
        if matrix is None or not isinstance(matrix, list) or \
            row_idx < 0 or row_idx >= len(matrix) or \
            matrix[row_idx] is None or not isinstance(matrix[row_idx], list) or \
            col_idx < 0 or col_idx >= len(matrix[row_idx]):
            return None
        return matrix[row_idx][col_idx]
    except (ValueError, IndexError, TypeError, AttributeError): # map(int,...)에서 ValueError 발생 가능
        return None