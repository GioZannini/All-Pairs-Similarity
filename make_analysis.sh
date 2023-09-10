#!/bin/bash

# path that indicate position of spark
path='/home/giogio/Scaricati/spark-3.4.0-bin-hadoop3/'

# command to start master
start_master=$path'sbin/start-master.sh'
# command to stop master
stop_master=$path'sbin/stop-master.sh'

# command to start one worker
start_worker=$path'sbin/start-worker.sh spark://GioGio:7077'
# command to stop worker/s
stop_worker=$path'sbin/stop-worker.sh'

# control right number of parameters in commmand line
if [ "$#" -ne 1 ]
then
	echo "This script requires only the name of .py file regards spark!"
else

    # execute command (start master)
    eval $start_master

    # number of workers
    for w in 1 2 4 8
    do
        # start w workers
        eval 'export SPARK_WORKER_INSTANCES=${w}; '$start_worker

        # different threshold
        for t in 0.15 0.50 0.85
        do
            # different dataset
            for d in 'scifact_subset_100_AUG' 'nfcorpus_subset_200_AUG' 'quora_subset_500_AUG'
            do
                # execute file.py given by command line 
                # this file.py takes itself 3 parameter by command line
                eval $path'bin/spark-submit --master spark://GioGio:7077 ./${1}' $d $t $w
            done
        done

        # stop workers
        eval $stop_worker
    done

    # stop master
    eval $stop_master
fi