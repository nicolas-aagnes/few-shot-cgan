# for datasetsize in 1000 5000 10000 50000
# do
#     for noiselevel in 0.01 0.05 0.1 0.15 0.2
#     do
#         python pretrain.py --dataset-size $datasetsize --noise-level $noiselevel
#     done
# done

# for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
# do
#     python pretrain.py --noise-level $noiselevel --dataset-size 25000 --niter 10
# done

# for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
# do
#     python pretrain.py --noise-level $noiselevel --dataset-size 10000 --niter 25
# done

datasetsize=50000
for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
do
    python train_refinement.py --dataset-size $datasetsize --noise-level $noiselevel --niter 5  --netG pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1914/netG.pth --netD pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1914/netD.pth 
done

datasetsize=10000
for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
do
    python train_refinement.py --dataset-size $datasetsize --noise-level $noiselevel --niter 25 --netG pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1946/netG.pth --netD pretrain/dataset_size\=$datasetsize\,noise_level\=$noiselevel/iteration1946/netD.pth 
done

# for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
# do
#     python train_refinement.py --dataset-size --noise-level $noiselevel --netG pretrain/dataset_size\=25000\,noise_level\=$noiselevel/iteration1914/netG.pth --netD pretrain/dataset_size\=25000\,noise_level\=$noiselevel/iteration1914/netD.pth 
# done


# for noiselevel in 0.0 0.05 0.1 0.2 0.3 0.5
# do
#     python train_refinement.py --noise-level $noiselevel --netG pretrain/dataset_size\=10000\,noise_level\=$noiselevel/iteration1914/netG.pth --netD pretrain/dataset_size\=10000\,noise_level\=$noiselevel/iteration1914/netD.pth 
# done