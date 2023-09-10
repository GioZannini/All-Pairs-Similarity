from Assignment3_modules import AllPairsSimilaritySparkVersionBase, ReadData
from pyspark.sql import SparkSession
import sys
from csv import writer
import time



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("WRONG NUMBER OF PARAMETERS")
    else:    
        # extract docs from dataset
        docs = ReadData.get(sys.argv[1])
        # create spark session
        spark = SparkSession.builder.appName("V1_Dataset_{}_|_Threshold_{}_|_Worker_{}".format(sys.argv[1], sys.argv[2], sys.argv[3])).getOrCreate()
        # start time
        start = time.time()
        # istance of obj for similarity pair
        similarity_pair = AllPairsSimilaritySparkVersionBase(docs, float(sys.argv[2]), spark.sparkContext)
        # take time
        preprocess_time = time.time() - start
        # start time
        start = time.time()
        # measure similarity between docs
        similarity_pair.measure_similarity()
        # take time
        process_time = time.time() - start
        # stop spark session
        spark.stop()
        # open existing CSV file in append mode
        with open('./spark_time.csv', 'a') as f:
            # get a writer object
            writer_object = writer(f)
            # write new row
            writer_object.writerow([sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), "v1", 
                                    preprocess_time, process_time, preprocess_time + process_time])
            # close the file object
            f.close()     