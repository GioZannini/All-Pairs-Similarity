import numpy as np


class ReadData:
    __FOLDER = "./Datasets/{}"

    @staticmethod
    def get(data_name, n_sample=None):
        # load data
        docs = np.load(ReadData.__FOLDER.format(data_name + '.npy'), allow_pickle=True).item()
        # extract dictionary with id_doc: only text of docs (without title and so on....)
        return {k: d["text"] for (k, d) in docs.items()}


