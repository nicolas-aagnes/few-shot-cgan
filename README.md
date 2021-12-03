# few-shot-cgan

# DONE:
- ``train_classifier.py`` trains an oracle classifier on the MNIST dataset -> This is what we will use for evaluation.
- ``train_cgan.ipynb`` trains a standard cGAN on MNIST.
- ``mnist_cnn.pth`` is the oracle classifier.

# TODOs:
- Pretrain classifier with CPC, and then just slightly tune it on the noisy target labels.
- Setup normal cGAN training loop for pretraining the discriminator and generator.
- Switch to finetuning the classifier with the generator and discriminator.


# CelebA TODOs:
- Train oracle attribute classifier on CelebA dataset.
- Make noisy celeba dataset. How to distribute the noisy labels?
- Use same training loop: first pretrain cgan, then pretrain the classifier on noisy labels, then jointly train them.

# Artifacts
1. celeba_oracle.pth was trained on Net with no pos weight and has accuracy of 90%
2. celeba_oracle.pth was trained on Net with pos weight and has accuracy of 87.8%