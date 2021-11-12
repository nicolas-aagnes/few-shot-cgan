# for datasetsize in 1000 5000 10000 50000
# do
#     for noiselevel in 0.01 0.05 0.1 0.15 0.2
#     do
#         python pretrain.py --dataset-size $datasetsize --noise-level $noiselevel
#     done
# done

for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
do
    python pretrain.py --noise-level $noiselevel
done