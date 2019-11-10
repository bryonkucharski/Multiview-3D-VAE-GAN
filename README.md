# Multiview-3D-VAE-GAN
Final Project for my graduate course CMPSCI674 - Intelligent Visual Computing at UMass. A popular method in 3D object classification is to use multiple 2D images (views) of the 3D object and pass through independent CNN networks. Recent research in generative 3D deep learning includes generating 3D voxelized objects using the VAE-GAN architecture (3D-VAE-GAN). In this project, I extend the existing 3D-VAE-GAN to include multiple views of 2D images for generating voxel based 3D objects. I found that my method had a lower average reconstruction loss on the test dataset.

The baseline code was taken from https://github.com/rimchang/3DGAN-Pytorch

3D-VAE-GAN architecture is introduced in the paper "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling".

3D-GAN - a 3D model is generated from a random latent vector <br/>
3D-VAE-GAN - A simple 2D image is encoded using a VAE and the corresponding 3D models are generated using a GAN.  <br/>
MV-3D-VAE-GAN - Multiple 2D images (multi view) are encoded using a VAE, pooled together, and the corresponding 3D models are generated using a GAN Mean pooling and max pooling are both options for each encoded image<br/>

# Results
![Image of Results](https://github.com/bryonkucharski/Multiview-3D-VAE-GAN/blob/master/src/3dvae_training.PNG)

# Model Architecture
3D-GAN</br>
![3DGAN](https://github.com/bryonkucharski/Multiview-3D-VAE-GAN/blob/master/src/gan.PNG)

</br>
3D-VAE-GAN</br>
![3DVAEGAN](https://github.com/bryonkucharski/Multiview-3D-VAE-GAN/blob/master/src/3dvaegan.PNG)

</br>
MV-3D-VAE-GAN</br>
![MV3DGAN](https://github.com/bryonkucharski/Multiview-3D-VAE-GAN/blob/master/src/MV3dvaegan.PNG)
