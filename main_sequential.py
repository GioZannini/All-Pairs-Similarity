from Assignment3_modules import TakeTimeInfo, ReadData, AllPairsSimilaritySeqVersion


DATASETS = ['scifact_subset_100_AUG', 'nfcorpus_subset_200_AUG', 'quora_subset_500_AUG']
THRESHOLD = [.15, .50, .85]

if __name__ == '__main__':
    # take time of experiment
    take_time = TakeTimeInfo(["data_name", "threshold", "n pairs"], 2, "sequential_time")
    # for each dataset
    for dataset in DATASETS:
        # for each threshold
        for t in THRESHOLD:
            print(f"------------Data: {dataset} | Threshold: {t}------------")
            # start time
            take_time.start_time()
            # extract docs from dataset
            docs = ReadData.get(dataset)
            # istance of obj for similarity pair
            similarity_pair = AllPairsSimilaritySeqVersion(docs, t)
            # take non parallel time
            take_time.split_time()
            # measure similarity between docs
            similarity_pair.measure_similarity()
            # take parallel time
            take_time.split_time()
            # number of pairs that outdo threshold
            pairs = similarity_pair.get_number_pairs()
            # save data in pandas dataframe
            take_time.insert_data([dataset, t, pairs])
    # save data as csv
    take_time.save_as_csv()
