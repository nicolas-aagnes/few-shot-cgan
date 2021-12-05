python train_slurm.py \
    --logdir $1 \
    --nz 128 \
    --ngf 1024 \
    --ndf 1024 \
    --batch_size 512 \
    --num_workers 16 \
    --beta1 0.1 \
    --lr 0.0001