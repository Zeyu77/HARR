import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter


def load_model(arch, code_length, num_cluster=30):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = model.classifier[:-3]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = ModelWrapper_res(model, 2048, code_length, num_cluster)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length, num_cluster):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

        self.head = nn.Sequential(nn.Linear(last_node, num_cluster),
            nn.Softmax(dim=1))


        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        else:
            feature = self.model(x)
            y = self.hash_layer(feature)
            return y

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


class ModelWrapper_res(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length, num_cluster):
        super(ModelWrapper_res, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

        self.head = nn.Sequential(nn.Linear(last_node, num_cluster),
            nn.Softmax(dim=1))


        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            batch_size = x.size()[0]
            return self.model(x).reshape(batch_size,2048)
        else:
            batch_size = x.size()[0]
            feature = self.model(x).reshape(batch_size,2048)
            y = self.hash_layer(feature)
            return y

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag