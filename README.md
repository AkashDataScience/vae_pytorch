[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Implement VAE Architecture from scratch
- Use MNIST and CIFAR10 dataset for training
- Give option for dataset

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]

## :open_file_folder: Files
- [**model.py**](model.py)
    - This file contains model architecture
- [**utils.py**](utils.py)
    - This file contains methods to plot metrics and results.
- [**train.py**](train.py)
    - This is the main file of this project
    - It uses function available in `model.py` and `utils.py`
    - It contrains functions to train and test model.

## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/vae_pytorch
```
2. Go inside folder
```
 cd vae_pytorch
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python train.py

# To train on MNIST dataset from jupyter notebook
%run train.py --mnist

```

## Usage 
Please refer to [ERA V2 Session 22](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-22)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/