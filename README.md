# HaHeAE: Learning Generalisable Joint Representations of Human Hand and Head Movements in Extended Reality
Project homepage: https://zhiminghu.net/hu25_haheae.


## Abstract
```
Human hand and head movements are the most pervasive input modalities in extended reality (XR) and are significant for a wide range of applications. 
However, prior works on hand and head modelling in XR only explored a single modality or focused on specific applications. 
We present HaHeAE - a novel self-supervised method for learning generalisable joint representations of hand and head movements in XR. 
At the core of our method is an autoencoder (AE) that uses a graph convolutional network-based semantic encoder and a diffusion-based stochastic encoder to learn the joint semantic and stochastic representations of hand-head movements. 
It also features a diffusion-based decoder to reconstruct the original signals. 
Through extensive evaluations on three public XR datasets, we show that our method 1) significantly outperforms commonly used self-supervised methods by up to 74.1% in terms of reconstruction quality and is generalisable across users, activities, and XR environments, 2) enables new applications, including interpretable hand-head cluster identification and variable hand-head movement generation, and 3) can serve as an effective feature extractor for downstream tasks. 
Together, these results demonstrate the effectiveness of our method and underline the potential of self-supervised methods for jointly modelling hand-head behaviours in extended reality.
```


## Environment:
Ubuntu 22.04
python 3.8+
pytorch 1.8.1


## Usage:
Step 1: Create the environment
```
conda env create -f ./environment/haheae.yaml -n haheae
conda activate haheae
```

Step 2: Follow the instructions at [Pose2Gaze][1] to process the datasets.


Step 3:  Set 'data_dir' in 'config.py' and 'main.py' for the processed datasets. Run 'train.sh' to evaluate the pre-trained models. If you want to train the model from scratch, you can remove the pre-trained models and uncomment the training command (the command with "mode" set to "train").

 
## Citation

```bibtex
@article{hu25haheae,
	author={Hu, Zhiming and Zhang, Guanhua and Yin, Zheming and Haeufle, Daniel and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={HaHeAE: Learning Generalisable Joint Representations of Human Hand and Head Movements in Extended Reality}, 
	year={2025}}
	
@article{hu24pose2gaze,
	author={Hu, Zhiming and Xu, Jiahui and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={Pose2Gaze: Eye-body Coordination during Daily Activities for Gaze Prediction from Full-body Poses}, 
	year={2024}}
```


## Acknowledgements
Our work is built on the codebase of [Diffusion Autoencoders][2] and [DisMouse][3]. Thanks to the authors for sharing their codes.

[1]: https://github.com/CraneHzm/Pose2Gaze
[2]: https://diff-ae.github.io/
[3]: https://git.hcics.simtech.uni-stuttgart.de/public-projects/DisMouse
