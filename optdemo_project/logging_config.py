import logging

# ANSI 색상 코드 정의
LOG_COLORS = {
    'DEBUG':    '\033[90m',   # 회색
    'INFO':     '\033[92m',   # 초록
    'WARNING':  '\033[93m',   # 노랑
    'ERROR':    '\033[91m',   # 빨강
    'CRITICAL': '\033[1;91m', # 굵은 빨강
}
RESET_COLOR = '\033[0m'

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        color = LOG_COLORS.get(levelname, '')
        message = super().format(record)
        return f"{color}{message}{RESET_COLOR}"

# 포맷 설정
debug_format = '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
info_format  = '[%(asctime)s] [%(levelname)s] %(message)s'

debug_formatter = ColoredFormatter(debug_format, datefmt='%Y-%m-%d %H:%M:%S')
info_formatter  = ColoredFormatter(info_format,  datefmt='%Y-%m-%d %H:%M:%S')

# DEBUG 전용 필터
class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        if isinstance(level, str):
            self.level = logging.getLevelName(level.upper())
        else:
            self.level = level

    def filter(self, record):
        return record.levelno == self.level

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # INFO 이상 출력 핸들러
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(info_formatter)
    logger.addHandler(info_handler)

    # DEBUG 레벨 전용 핸들러
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(LevelFilter(logging.DEBUG))
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)