# GANs Study with PyTorch, Cat Face Generation

This project is a study on Generative Adversarial Networks (GANs). It includes a Jupyter notebook for debugging, a shell script for downloading cat images, and a Python package named [`gan`](command:_github.copilot.openRelativePath?%5B%22gan%22%5D "gan") containing various modules.

## File Structure

```
GAN_debugging.ipynb
download_cat.sh
gan/
	.ipynb_checkpoints/
		train-checkpoint.py
		utils-checkpoint.py
	__pycache__/
	losses.py
	models.py
	spectral_normalization.py
	train.py
	utils.py
gan_samples/
```

## Description

- [`GAN_debugging.ipynb`](command:_github.copilot.openRelativePath?%5B%22GAN_debugging.ipynb%22%5D "GAN_debugging.ipynb"): A Jupyter notebook for debugging the GAN model.
- [`download_cat.sh`](command:_github.copilot.openRelativePath?%5B%22download_cat.sh%22%5D "download_cat.sh"): A shell script for downloading cat images.
- [`gan/train.py`](command:_github.copilot.openRelativePath?%5B%22gan%2Ftrain.py%22%5D "gan/train.py"): The main training script for the GAN model.
- [`gan/utils.py`](command:_github.copilot.openRelativePath?%5B%22gan%2Futils.py%22%5D "gan/utils.py"): Contains utility functions used across the project.
- [`gan/models.py`](command:_github.copilot.openRelativePath?%5B%22gan%2Fmodels.py%22%5D "gan/models.py"): Defines the GAN models.
- [`gan/losses.py`](command:_github.copilot.openRelativePath?%5B%22gan%2Flosses.py%22%5D "gan/losses.py"): Defines the loss functions used in training the GAN.
- [`gan/spectral_normalization.py`](command:_github.copilot.openRelativePath?%5B%22gan%2Fspectral_normalization.py%22%5D "gan/spectral_normalization.py"): Implements spectral normalization, a technique for stabilizing the training of the GAN.

## How to Run

1. Download the cat images by running the shell script:

```sh
./download_cat.sh
```

2. Train the GAN model:

```sh
python gan/train.py
```

3. Debug the GAN model using the Jupyter notebook:

```sh
jupyter notebook GAN_debugging.ipynb
```

## Output

The output of the GAN model will be saved in the [`gan_samples/`](command:_github.copilot.openRelativePath?%5B%22gan_samples%2F%22%5D "gan_samples/") directory.