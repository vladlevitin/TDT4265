from re import M
from xml.dom import InvalidCharacterErr
from xml.etree.ElementTree import tostring
from numpy import pad
import torch
from typing import Tuple, List
import torchvision.transforms as transforms


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # Store the different architechtures in a list
        self.feature_extractors = torch.nn.ModuleList([
            
            torch.nn.Sequential(
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=128, out_channels=output_channels[0], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU()
            ),

            torch.nn.Sequential(
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=output_channels[0], out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),


                torch.nn.Conv2d(in_channels=128, out_channels=output_channels[1], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU()
            ),

             torch.nn.Sequential(
                 torch.nn.ReLU(),

                 torch.nn.Conv2d(in_channels=output_channels[1], out_channels=256, kernel_size=3, padding=1),
                 torch.nn.BatchNorm2d(256),
                 torch.nn.ReLU(),
                 
                 torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                 torch.nn.BatchNorm2d(256),
                 torch.nn.ReLU(),
                 
                 torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                 torch.nn.BatchNorm2d(256),
                 torch.nn.ReLU(),

                 torch.nn.Conv2d(in_channels=256, out_channels=output_channels[2], kernel_size=3, padding=1, stride=2),
                 torch.nn.ReLU()
             ),

            torch.nn.Sequential(
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=output_channels[2], out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=128, out_channels=output_channels[3], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU()
            ),

            torch.nn.Sequential(
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=output_channels[3], out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=128, out_channels=output_channels[4], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU()
            ),

            torch.nn.Sequential(
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=output_channels[4], out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),

                torch.nn.Conv2d(in_channels=128, out_channels=output_channels[5], kernel_size=3, padding=0, stride=1),
                torch.nn.ReLU()
            )
        ])

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out = x
        
        for extractor in self.feature_extractors:
            out = extractor(out)
            out_features.append(out)
        
        # Send through first layers
        #extractor = self.feature_extractors[0]
        #out_feature = extractor(x)
        #out_features.append(out_feature)
        
        # Iterate through next layers and send through
        #for idx, extractor in enumerate(self.feature_extractors[1:]):
            #out_feature = extractor(out_features[idx])
            #out_features.append(out_feature)
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

