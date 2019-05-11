# Multiview-3D-VAE-GAN
Final Project for my graduate course CMPSCI674 - Intelligent Visual Computing at UMass

The baseline code was taken from https://github.com/rimchang/3DGAN-Pytorch

Multiview extension of the 3D-VAE-GAN architecture introduced in the paper "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling".

3D-GAN - a 3D model is generated from a random latent vector
3D-VAE-GAN - A simple 2D image is encoded using a VAE and the corresponding 3D models are generated using a GAN. 
MV-3D-VAE-GAN - Multiple 2D images are encoded using a VAE, pooled together, and the corresponding 3D models are generated using a GAN. Mean pooling and max pooling are both options for each encoded image.

# Results

