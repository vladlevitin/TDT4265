from cgi import test
import pathlib
import matplotlib.pyplot as plt
import torch
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import torchvision.transforms as transforms

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class emilNet(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.num_classes = 10



        # Define the convolutional and fully connected layers
        self.feature_extractor = nn.Sequential(
            # Transformations              
            transforms.ColorJitter(),

            # Conv layer 1 
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=32),
            nn.Tanh(),
            nn.Dropout(p=0.2),

            # Conv layer 2 
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            
            # Conv layer 2 
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Conv layer 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
             
            # Conv layer 4
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Conv layer 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Fully connected layers
            nn.Flatten(start_dim=1, end_dim=3), # Flatten in last three dimensions, [64, 2048]
            nn.Linear(in_features= 256 * 4 * 4, out_features=512), # [64, 128]
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256), # [64, 128]
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128), # [64, 128]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10), # [64, 10]
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(13)
    epochs = 10
    batch_size = 64
    learning_rate = 0.5e-1

    early_stop_count = 7
    dataloaders = load_cifar10(batch_size)
    model = emilNet(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "emilNet")
    
    trainer.test_loss()


if __name__ == "__main__":
    main()
