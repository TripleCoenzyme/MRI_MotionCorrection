# MRI_MotionCorrection
A U-Net based end-to-end MRI motion correction reimplementation in pytorch. With/Without GAN version is available.

The basic idea and the motion simulation is based on the MRM article [Conditional generative adversarial network for 3D rigid-body motion correction in MRI](https://onlinelibrary.wiley.com/doi/10.1002/mrm.27772) and the Github code [MoCo_cGAN](https://github.com/pjohnson519/MoCo_cGAN). And there is a lot of code reused from [ResNet50-Unet](https://github.com/TripleCoenzyme/ResNet50-Unet).

The scheduler position is updated for higher version (higher than 1.1.0) of pytorch. Please check out the [official warning](https://pytorch.org/docs/stable/optim.html#:~:text=Prior%20to%20PyTorch%201.1.0%2C%20the%20learning%20rate%20scheduler,the%20first%20value%20of%20the%20learning%20rate%20schedule).

This project can also be used as a template for any U-Net based task and (patch) GAN-based task. 
