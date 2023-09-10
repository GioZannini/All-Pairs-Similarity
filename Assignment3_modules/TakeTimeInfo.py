import pandas as pd
import time


class TakeTimeInfo:
    __COL_NAME = "Split_{}"

    def __init__(self, col: list, splits: int, name_csv: str):
        self.name_csv = name_csv
        self.table = pd.DataFrame(columns=col + [TakeTimeInfo.__COL_NAME.format(i+1) for i in range(splits)] + ["Total time"])
        self.time = None
        self.time_vector = None

    def start_time(self):
        self.time_vector = []
        self.time = time.time()

    def split_time(self):
        # save elapsed time
        self.time_vector.append(time.time() - self.time)
        # change actual time
        self.time = time.time()

    def insert_data(self, data: list):
        # find total time
        self.time_vector.append(sum(self.time_vector))
        pd_tmp = pd.DataFrame([data + self.time_vector],
                              columns=self.table.columns)
        self.table = pd.concat([self.table, pd_tmp])

    def save_as_csv(self):
        self.table.to_csv(self.name_csv+".csv", index=False)



