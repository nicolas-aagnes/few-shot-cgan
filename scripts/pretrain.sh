for datasetsize in 1000 5000 10000 50000
do
    for noiselevel in 0.01 0.05 0.1 0.15 0.2
    do
        python pretrain.py --dataset-size $datasetsize --noise-level $noiselevel
    done
done