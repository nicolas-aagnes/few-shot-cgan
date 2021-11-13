datasetsize=50000
for noiselevel in 0.75 0.5 0.25
do
    for entropy in 0.05 1.0 2.0
    do
        CUDA_VISIBLE_DEVICES=0 python train_pseudo.py \
            --dataset-size $datasetsize \
            --noise-level $noiselevel \
            --entropy $entropy \
            --exp-name "entropy"
    done
done
