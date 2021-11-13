datasetsize=50000
for noiselevel in 0.5 0.3 0.2 # 0.1 0.05 0.0
do
    for entropy in 0.05 0.1 0.5 # 1.0 1.5 2.0 2.3
    do
        CUDA_VISIBLE_DEVICES=0 python train_pseudo.py \
            --dataset-size $datasetsize \
            --noise-level $noiselevel \
            --entropy $entropy
    done
done
