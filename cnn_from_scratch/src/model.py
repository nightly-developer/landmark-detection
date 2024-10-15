import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # activation function 
        self.activation = torch.nn.functional.relu
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        
        # Cov. 1
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3, padding= 1) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchNorm1 = nn.BatchNorm2d(32)

        
        # Cov. 2
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, padding= 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchNorm2 = nn.BatchNorm2d(64)
        
        # Cov. 3
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, padding= 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batchNorm3 = nn.BatchNorm2d(128)

        # Cov. 4
        self.conv4 = nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, padding= 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batchNorm4 = nn.BatchNorm2d(256)
        
        # Flattening
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*14*256, 512)
        self.batchNorm5 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, num_classes)


        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)        
        
        x = self.pool1(self.activation(self.batchNorm1(self.conv1(x)))) 
        x = self.pool2(self.activation(self.batchNorm2(self.conv2(x))))
        x = self.pool3(self.activation(self.batchNorm3(self.conv3(x)))) 
        x = self.pool4(self.activation(self.batchNorm4(self.conv4(x)))) 

        x = self.flatten(x)
        
        x = self.activation(self.batchNorm5(self.dropout(self.fc1(x))))
        
        x = self.fc2(x)
                
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
