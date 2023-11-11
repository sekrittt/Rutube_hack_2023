from time import process_time

logging = True

def disable_logging():
    global logging
    logging = False

def timeit(func):
    """Декоратор для замера времени выполнения функции"""

    def wrapper(*args, **kwargs):
        global logging
        start_time = process_time()
        result = func(*args, **kwargs)
        end_time = process_time()
        if logging:
            print(
                f"Время выполнения функции '{func.__name__}': {end_time - start_time} секунд"
            )
        return result

    return wrapper
