import torch.nn as nn
import math

def reshape_model(model: nn.Module, channels: int, n_classes: int) -> nn.Module:
    """Reshapes input and output for the model. Note that currently this does not detect models
    which are not suited for the given input data size!

    Args:
        model (nn.Module): PyTorch model
        channels (int): input channels
        n_classes (int): number of classes

    Raises:
        NotImplementedError: reshape for model is not implemented

    Returns:
        nn.Module: model with reshaped input and output layer
    """
    if model.__class__.__name__ == 'ResNet':
        model.conv1 = nn.Conv2d(channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=1, padding=3)
        model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
    elif model.__class__.__name__ == 'VGG':
        model.features[0] = nn.Conv2d(
            channels, 
            model.features[0].out_channels, 
            kernel_size=model.features[0].kernel_size, 
            stride=model.features[0].stride, 
            padding=model.features[0].padding,
            padding_mode=model.features[0].padding_mode,
            dilation=model.features[0].dilation,
            groups=model.features[0].groups,
            bias=True if model.features[0].bias is not None else False
        )
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, 
            n_classes, 
            bias=True if model.classifier[-1].bias is not None else False
        )
    elif model.__class__.__name__  == 'SqueezeNet':
        model.features[0] = nn.Conv2d(
            channels, 
            model.features[0].out_channels, 
            kernel_size=model.features[0].kernel_size, 
            stride=model.features[0].stride, 
            padding=model.features[0].padding,
            padding_mode=model.features[0].padding_mode,
            dilation=model.features[0].dilation,
            groups=model.features[0].groups,
            bias=True if model.features[0].bias is not None else False
        )
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels, 
            n_classes, 
            kernel_size=model.classifier[1].kernel_size, 
            stride=model.classifier[1].stride, 
            padding=model.classifier[1].padding,
            padding_mode=model.classifier[1].padding_mode,
            dilation=model.classifier[1].dilation,
            groups=model.classifier[1].groups,
            bias=True if model.classifier[1].bias is not None else False
        )
    elif model.__class__.__name__  == 'DenseNet':
        model.features[0] = nn.Conv2d(
            channels, 
            model.features[0].out_channels, 
            kernel_size=model.features[0].kernel_size, 
            stride=model.features[0].stride, 
            padding=model.features[0].padding,
            padding_mode=model.features[0].padding_mode,
            dilation=model.features[0].dilation,
            groups=model.features[0].groups,
            bias=True if model.features[0].bias is not None else False
        )
        model.classifier = nn.Linear(
            model.classifier.in_features, 
            n_classes, 
            bias=True if model.classifier.bias is not None else False
        )
    elif model.__class__.__name__  == 'Inception3':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(
            channels, 
            model.Conv2d_1a_3x3.conv.out_channels, 
            kernel_size=model.Conv2d_1a_3x3.conv.kernel_size, 
            stride=model.Conv2d_1a_3x3.conv.stride, 
            padding=model.Conv2d_1a_3x3.conv.padding,
            padding_mode=model.Conv2d_1a_3x3.conv.padding_mode,
            dilation=model.Conv2d_1a_3x3.conv.dilation,
            groups=model.Conv2d_1a_3x3.conv.groups,
            bias=True if model.Conv2d_1a_3x3.conv.bias is not None else False
        )
        model.fc = nn.Linear(
            model.fc.in_features, 
            n_classes, 
            bias=True if model.fc.bias is not None else False
        )
    elif model.__class__.__name__  == 'GoogLeNet':
        model.conv1.conv = nn.Conv2d(
            channels, 
            model.conv1.conv.out_channels, 
            kernel_size=model.conv1.conv.kernel_size, 
            stride=model.conv1.conv.stride, 
            padding=model.conv1.conv.padding,
            padding_mode=model.conv1.conv.padding_mode,
            dilation=model.conv1.conv.dilation,
            groups=model.conv1.conv.groups,
            bias=True if model.conv1.conv.bias is not None else False
        )
        model.fc = nn.Linear(
            model.fc.in_features, 
            n_classes, 
            bias=True if model.fc.bias is not None else False
        )
    elif model.__class__.__name__  == 'ShuffleNetV2':
        model.conv1[0] = nn.Conv2d(
            channels, 
            model.conv1[0].out_channels, 
            kernel_size=model.conv1[0].kernel_size, 
            stride=model.conv1[0].stride, 
            padding=model.conv1[0].padding,
            padding_mode=model.conv1[0].padding_mode,
            dilation=model.conv1[0].dilation,
            groups=model.conv1[0].groups,
            bias=True if model.conv1[0].bias is not None else False
        )
        model.fc = nn.Linear(
            model.fc.in_features, 
            n_classes, 
            bias=True if model.fc.bias is not None else False
        )
    elif model.__class__.__name__  == 'MobileNetV2':
        model.features[0][0] = nn.Conv2d(
            channels, 
            model.features[0][0].out_channels, 
            kernel_size=model.features[0][0].kernel_size, 
            stride=model.features[0][0].stride, 
            padding=model.features[0][0].padding,
            padding_mode=model.features[0][0].padding_mode,
            dilation=model.features[0][0].dilation,
            groups=model.features[0][0].groups,
            bias=True if model.features[0][0].bias is not None else False
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 
            n_classes, 
            bias=True if model.classifier[1].bias is not None else False
        )
    elif model.__class__.__name__  == 'MNASNet':
        model.layers[0] = nn.Conv2d(
            channels, 
            model.layers[0].out_channels, 
            kernel_size=model.layers[0].kernel_size, 
            stride=model.layers[0].stride, 
            padding=model.layers[0].padding,
            padding_mode=model.layers[0].padding_mode,
            dilation=model.layers[0].dilation,
            groups=model.layers[0].groups,
            bias=True if model.layers[0].bias is not None else False
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 
            n_classes, 
            bias=True if model.classifier[1].bias is not None else False
        )
    elif model.__class__.__name__  in ['MLP', 'CNN']:
        pass
    else:
        raise NotImplementedError
    return model