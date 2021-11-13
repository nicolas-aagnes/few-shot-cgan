datasetsize=50000
for noiselevel in 0.2 0.3 0.5
do
    python train_refinement.py --dataset-size $datasetsize --noise-level $noiselevel --niter 5  --netG pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1914/netG.pth --netD pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1914/netD.pth 
done

# datasetsize=10000
# for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
# do
#     python train_refinement.py --dataset-size $datasetsize --noise-level $noiselevel --niter 25 --netG pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1946/netG.pth --netD pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1946/netD.pth 
# done