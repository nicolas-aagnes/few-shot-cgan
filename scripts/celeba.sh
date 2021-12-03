# rm -rf runs_celeba
python train_cgan.py \
    --logdir $1 \
    --nz 128 \
    --ngf 1024 \
    --ndf 1024 \
    --batch-size 512 \
    --num-workers 16 \
    --beta1 0.1 \
    --lr 0.0001