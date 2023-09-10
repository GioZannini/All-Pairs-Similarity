#--------------------------------------------------------------------------------------------------------------
# Used to increment similarity between docs, replicate some entries in dataset and 
# modify a bit the content of their text to achieve a better similarity (close to 1)
# 
# From command line takes:
# name dataset | percentage of augmenting (between 0 and 1) | percentage of max number of word eliminated from text
#---------------------------------------------------------------------------------------------------------------

import numpy as np
import sys

# remove at most p_max_change percent of text in random way
def change_a_bit_of_content(d, p_max_change):
    # extract text and transform in list
    list_of_s = np.array(d['text'].split())
    dim = list_of_s.shape[0]
    # find index to remove
    remove_idx = np.random.choice(dim, size=np.random.randint(0, 1 if int(dim * p_max_change) == 0 else int(dim * p_max_change)), replace=False)
    # remove specific element from list
    list_of_s = np.delete(list_of_s, remove_idx)
    d['text'] = " ".join(list_of_s)
    return d


if __name__ == '__main__':
    FOLDER = "./../Datasets/{}"
    # load data
    docs = np.load(FOLDER.format(sys.argv[1] + '.npy'), allow_pickle=True).item()
    # max keys
    try:
        max_keys = max(np.array(list(docs.keys()), dtype=int))
    except ValueError:
        max_keys = 0
    # trasform dictionary in list of pairs
    tmp_docs = list(docs.items())
    # extract percentage of docs that I want for incrementing similarity
    augmented_docs = tmp_docs[:int(float(sys.argv[2])*len(tmp_docs))]
    # change keys, in this way I avoid duplicates
    augmented_docs = [(str(max_keys+1+e), change_a_bit_of_content(v, float(sys.argv[2]))) for e, (k, v) in enumerate(augmented_docs)]
    # concatenate 2 lists
    final_augmented = dict(tmp_docs + augmented_docs)
    # save dictionary as file
    np.save(FOLDER.format(sys.argv[1] + '_AUG.npy'), final_augmented) 