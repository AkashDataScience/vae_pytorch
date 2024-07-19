import torch
import argparse
from model import VAE
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch VAE Training')
    parser.add_argument('--epochs', default=30, type=int, help="Number of training epochs e.g 25")
    parser.add_argument('--batch_size', default=32, type=int, help="Number of images per batch e.g. 256")
    parser.add_argument('--mnist', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

def start_training(num_epochs, model, device, data_loader):
        trainer = pl.Trainer(accelerator = device, max_epochs=num_epochs)
        trainer.fit(model, data_loader)

def main():
    args = get_args()

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    if args.mnist:
         datamodule = MNISTDataModule('.', batch_size=args.batch_size)
         model = VAE(input_height=28)
    else:
         datamodule = CIFAR10DataModule('.', batch_size=args.batch_size)
         model = VAE()
    
    device = "cuda" if cuda else "cpu"

    start_training(args.epochs, model, device, datamodule)

if __name__ == "__main__":
     main()