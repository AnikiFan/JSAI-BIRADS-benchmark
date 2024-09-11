
# import logging


# # 自定义过滤器类，用于过滤指定的日志消息
# class MultiMessageFilter(logging.Filter):
#     def __init__(self):
#         super().__init__()
#         self.filtered_messages = [
#             "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros.",
#             "STREAM b'IHDR'",
#             "STREAM b'IDAT'"
#         ]

#     def filter(self, record):
#         # 如果消息与任意一个过滤消息匹配，则过滤掉
#         return not any(msg in record.getMessage() for msg in self.filtered_messages)


# # 创建 logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# # 创建控制台处理器
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# # 添加自定义过滤器
# multi_message_filter = MultiMessageFilter()
# console_handler.addFilter(multi_message_filter)

# # 将处理器添加到 logger
# logger.addHandler(console_handler)



import logging

# 自定义过滤器类，用于过滤指定的日志消息
class MultiMessageFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.filtered_messages = [
            "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros.",
            "STREAM b'IHDR'",
            "STREAM b'IDAT'"
        ]

    def filter(self, record):
        # 如果消息与任意一个过滤消息匹配，则过滤掉
        return not any(msg in record.getMessage() for msg in self.filtered_messages)


# 创建 logger 和配置函数
def setup_custom_logger():
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 添加自定义过滤器
    multi_message_filter = MultiMessageFilter()
    console_handler.addFilter(multi_message_filter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)

    return logger