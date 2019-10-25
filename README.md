# OnGansAndGMMs

The project is based on the paper: https://arxiv.org/abs/1805.12462

The proposal for the project may be found here: https://www.overleaf.com/read/xgkqshgyqngd (Richardson  and  Weiss2018)

## Proposal summary

For convinience main milestones and list of experiments to be conducted is listed below (note that this list maybe changed in future):

Here we will briefly describe the goals of the project as well as important milestones.

At first, we want to reproduce the key experiments of Richardson  and  Weiss2018:
- Implement Voronoi cells based statistic test
- Implement Mixture of Factor Analysers (MFA) for data distribution modeling
- Compare MFA and GANS using statistic test proposed in paper, IS and FID on CelebA dataset.

There is also a number of interesting experiments, which can be conducted:
- Reproduce visual insights into the mode collapse problem
- Compare the ability of latent variables of GANs and GMMs to describe data manifold.
- Replace Mixture of FA with Mixture of PPCAs and compare the difference

## Team members
- Konstantin Pakulev
- Alisa Alenicheva
- Olga Tsimboy

## MFA
### Training
To train MFA from scratch you need to specify ```dataset_root``` parameter in [mfa_train.sh](mfa_train.sh). The root folder should contain folder with aligned and cropped images and train/val/test partitioning. Refer [dataset page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for more info.

### Evaluation
Download the [pre-trained FA based GMM model](https://drive.google.com/open?id=1CdSbcTV-zK55vVi3tZ-tXy0zGynZkIwp) or [pre-trained PPCA based GMM model](https://drive.google.com/open?id=1J792PyhOpSE2UsKEFTR1RhSw3xF1x6mO) or train them yourself.<br>
Edit ```path_to_fa_model``` and ```path_to_ppca_model``` parameters in [mfa_eval.ipynb](notebooks/mfa_eval.ipynb) and run it.


## DCGAN
### Training
DCGAN can pe trained by running `python3 dcgan_train.py`, there are various parameters of the model to variate: `python3 dcgan_train.py -h`


### Evaluation
Download the [pre-trained FA based DCGAN model](https://drive.google.com/open?id=1l0qgxEsefqVQpNG_REb3CY8h-8KvgjeT).
Edit ```data_path``` and ```LOG_PATH``` parameters in [dcgan_eval.ipynb](notebooks/dcgan_eval.ipynb) and run it.

## pix2pix
## Training 
For training pix2pix CGAN from scratch in [pix2pix_train.sh](pix2pix_train.sh) specify two parameters: ```dataset_root``` (which points to CelebA) and ```mfa_model_path``` (which points to trained MFA model). Alternatively, you can download pre-trained pix2pix model from [here](https://drive.google.com/open?id=1YIGKFxAnNnxmueZihugNzlk7v0i7Bp-I).

## Evaluation
Launch pix2pix [evaluation notebook](notebooks/pix2pix_eval.ipynb) and replace variable responsible for CelebA dataset path and trained pix2pix model path.



