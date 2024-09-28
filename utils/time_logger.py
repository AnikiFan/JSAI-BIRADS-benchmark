import time
import logging
def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        logging.info(f"Function '{func.__name__}' started at: {start_time_readable}")

        result = func(*args, **kwargs)

        end_time = time.time()
        end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        elapsed_time = end_time - start_time

        logging.info(f"Function '{func.__name__}' ended at: {end_time_readable}")
        logging.info(f"Elapsed time: {elapsed_time:.4f} seconds")

        return result
    return wrapper