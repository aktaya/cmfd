import time
import pickle
import uuid

class Logger:
    def __init__(self):
        self.logs = []

    def show_all(self):
        for l in self.logs:
            print(l)

    def show_last(self):
        print(self.logs[-1], flush=True)

    def __call__(self, msg):
        str_time = time.strftime("[%Y/%m/%d %H:%M:%S] ", time.localtime())
        _msg = str_time + msg
        self.logs.append(_msg)
        self.show_last()

    @staticmethod
    def strarray(ar, precision=3):
        fmt = "{:." + str(precision) + "f}"
        return ",".join([fmt.format(_) for _ in ar])

class PickleUtil:
    @staticmethod
    def save_data(dat, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dat, f)

    @staticmethod
    def load_data(filename):
        with open(filename, 'rb') as f:
            dat = pickle.load(f)
        return dat

class FileNameGenerator:
    def __init__(self, results_dir="results", debug=False):
        self.debug = debug
        self.prefix = results_dir + "/"

    def generate(self):
        if self.debug:
            return "debug/" + str(uuid.uuid4())
        else:
            return self.prefix + str(uuid.uuid4())

