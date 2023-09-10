#---------------------------------------------------------------------------------------------
# Used to create a random subset from dataset
# 
# From command line takes:
# name dataset | dimension subset
#---------------------------------------------------------------------------------------------

import numpy as np
import sys

if __name__ == '__main__':
    FOLDER = "./../Datasets/{}"
    # load data
    docs = np.load(FOLDER.format(sys.argv[1] + '.npy'), allow_pickle=True).item()
    # extract random subset of dimension n_sample
    
    tmp_docs = np.array(list(docs.items()))
    rand_subset = np.random.choice(tmp_docs.shape[0], size=int(sys.argv[2]), replace=False)
    docs_sample = dict(tmp_docs[rand_subset])
    # save dictionary as file
    np.save(FOLDER.format(sys.argv[1] + f'_subset_{sys.argv[2]}.npy'), docs_sample) 
    