import time
from datetime import datetime


class Timer(object):
    def __init__(self, accuracy=2, dump_to_file=False, date_format='%d/%m/%Y %H:%M:%S', message=''):
        self.accuracy = accuracy
        self.dump_to_file = dump_to_file
        self.date_format = date_format
        self.message = message

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = function(*args, **kwargs)
            end_time = time.time()
            time_diff = (end_time - start_time)

            timing_result = f'Execution of method {function.__name__} took exactly: {round(time_diff, self.accuracy)}s.'
            time_now = datetime.now()
            formatted_date = time_now.strftime(self.date_format)
            log_content = f'[{formatted_date} - {self.message}]: {timing_result}'

            if self.dump_to_file:
                with open('./logs/time.log', mode='a') as time_log:
                    time_log.write(log_content + "\n")

            print(log_content)
            return result

        return wrapper
