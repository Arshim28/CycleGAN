# CycleGAN Image-to-Image Translation

CycleGAN is a deep learning model for image-to-image translation without paired data. It can transform images from one domain to another, such as turning horse images into zebra images.

## Directory Structure

- `data.py`: Script for loading Datasets and preparing DataLoaders

- `config.json`: Configuration file where you can specify hyperparameters and training settings.

- `main.py`: Main script to train and run the CycleGAN model.

- `model.py`: Contains the architecture of the Generator and Discriminator.

- `train.py`: Contains functions for loading the configuration and training the model.

## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/cyclegan-project.git
   cd cyclegan-project
