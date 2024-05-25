# VQGAN
This is an implementation of Vector Quantized Generative Adversarial Networks (VQGAN) which is a generative model for images which was proposed in Taming Transformers for High-Resolution Image Synthesis. The first stage is to train a variation on VQVAE which uses a discriminator and perceptual loss to retain perceptual quality. The second stage employs the uses a transformer to perform sequence perdiction on the quantized latent representation of the image, after it has been trained the transformer can be prompted with an SOS token to generate a new image.

<hr>

![VQGAN_ARCH](https://github.com/SJ1727/VQGAN/assets/114866209/b287bd86-12c2-405f-9db8-3cb862e7be9d)

<hr>
