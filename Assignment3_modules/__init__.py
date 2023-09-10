from .AllPairsSimilarity import AllPairsSimilaritySeqVersion, AllPairsSimilaritySparkVersionBase, AllPairsSimilaritySparkVersion2, AllPairsSimilaritySparkVersion3
from .ReadData import ReadData
from .TakeTimeInfo import TakeTimeInfo

# take 2 dictionary composed by [(doc_id_1: (doc_id_2, cosine)), ...] 
# and control if they are equal
def verify_equality_between_models(d1:dict, d2:dict, n_digit=10):
    # control if the keys of dicitonaries are the same
    print(f"Key are equal: {sorted(list(d1.keys())) == sorted(list(d2.keys()))}")
    # count number of mistake
    m_i = 0
    # control if each value for each key is equal for both models
    for k in d1.keys():
        s1 = sorted(d1[k])
        s2 = sorted(d2[k])
        s1 = [(d, round(v, n_digit)) for (d, v) in s1]
        s2 = [(d, round(v, n_digit)) for (d, v) in s2]
        if s1 != s2:
            # key with different elements in values
            print(f"Errors in key {k}")
            m_i+=1
    print(f"Total mistake: {m_i}")

