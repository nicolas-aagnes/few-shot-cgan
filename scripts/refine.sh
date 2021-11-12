for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
do
    python train_refinement.py --dataset-size 50000 --noise-level $noiselevel --netG pretrain/dataset_size\=50000\,noise_level\=$noiselevel/iteration1914/netG.pth --netD pretrain/dataset_size\=50000\,noise_level\=$noiselevel/iteration1914/netD.pth 
done