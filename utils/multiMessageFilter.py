import logging


# 自定义过滤器类，用于过滤指定的日志消息
class MultiMessageFilter(logging.Filter):
    filtered_messages = [
        "STREAM",
        "b'tIME'",
        "iCCP",
        "Compression method",
        "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros.",
    ]

    def __init__(self):
        super().__init__()

    def filter(self, record):
        # 如果消息与任意一个过滤消息匹配，则过滤掉
        return not any(record.getMessage().startswith(msg) for msg in MultiMessageFilter.filtered_messages)

    def setup(self):
        # 创建 logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # 添加自定义过滤器
        console_handler.addFilter(self)
        # 将处理器添加到 logger
        logger.addHandler(console_handler)
