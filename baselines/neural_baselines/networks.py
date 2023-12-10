import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights


def generate_net(model_name: str, num_outputs: int, input_shape: tuple | list) -> \
        tuple[nn.Module, transforms.Compose, transforms.Compose]:
    """Generate a neural network model, moving it to the right device.

        :param model_name: Name of the network ('mlp', 'resnet50', ...).
        :param num_outputs: Number of output neurons.
        :param input_shape: Shape of the input data (c, h, w).
        :returns: The neural network model; the training transforms; the val/test (eval) transforms.
    """

    assert input_shape is not None and isinstance(input_shape, (tuple, list)), \
        "The input_shape field must be associated to a tuple or list with the shape of the input data (e.g. (1,32,32))"
    assert len(input_shape) == 3, \
        "Invalid 'input_shape' options, it must be a tuple like (c, h w)."
    assert num_outputs > 0, \
        "Invalid number of output units, it must be > 0."

    # unpacking
    c, h, w = input_shape

    # transformations (keeping original resolution)
    common_train_transforms = transforms.Compose([
        transforms.RandomRotation(15, fill=127),  # filling with the background color, 127
        transforms.RandomResizedCrop((h, w), scale=(0.95, 1.05), ratio=(0.95, 1.05)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.497, 0.497, 0.497], std=[0.065, 0.065, 0.065]),
    ])

    common_eval_transforms = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.497, 0.497, 0.497], std=[0.065, 0.065, 0.065]),
    ])

    # transformations (changing original resolution to 224x224)
    pretrained_resnet_like_train_transforms = transforms.Compose([
        transforms.RandomRotation(15, fill=127),  # filling with the background color, 127
        transforms.RandomResizedCrop(224, scale=(0.95, 1.05), ratio=(0.95, 1.05)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pretrained_resnet_like_eval_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # MLP network
    if model_name == 'mlp':
        input_size = int(np.prod(input_shape))

        net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Tanh(),
            nn.Linear(100, num_outputs)
        )

        train_transforms = common_train_transforms
        train_transforms.transforms.append(torch.flatten)
        eval_transforms = common_eval_transforms
        eval_transforms.transforms.append(torch.flatten)

    # CNN
    elif model_name == 'cnn':

        net = nn.Sequential(
                nn.Conv2d(c, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d(output_size=(5, 5)),  # it forces the output resolution to be 5x5
                nn.Flatten(),
                nn.Linear(256 * 5 * 5, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_outputs)
        )

        train_transforms = common_train_transforms
        eval_transforms = common_eval_transforms

    # ResNet50 (trained from scratch)
    elif model_name == 'resnet50':
        net = resnet50()

        train_transforms = common_train_transforms
        eval_transforms = common_eval_transforms

    # ResNet50 (pretrained backbone, training head only)
    elif model_name == 'resnet50_head_only':

        # ugly hack to fix SSL related issues
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        net = resnet50(weights=ResNet50_Weights.DEFAULT)

        for parameter in net.parameters():
            parameter.requires_grad = False

        net.fc = nn.Linear(in_features=2048, out_features=num_outputs)

        train_transforms = pretrained_resnet_like_train_transforms
        eval_transforms = pretrained_resnet_like_eval_transforms

    # ViT Base 16 (pretrained backbone, training head only)
    elif model_name == 'vit_head_only':

        # ugly hack to fix SSL related issues
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        for parameter in net.parameters():
            parameter.requires_grad = False

        net.heads = nn.Linear(in_features=768, out_features=num_outputs)

        train_transforms = pretrained_resnet_like_train_transforms
        eval_transforms = pretrained_resnet_like_eval_transforms
    else:
        raise ValueError(f"Unknown model: {model_name}")

    net.register_buffer("decision_thresholds", 0.5 * torch.ones(num_outputs))
    return net, train_transforms, eval_transforms


def save_net(net: tuple[nn.Module] | list[nn.Module] | nn.Module, filename: str) -> None:
    """Save a network to file.

        :param net: A Pytorch net or a list/tuple of nets.
        :param filename: The destination file.
    """

    if not isinstance(net, (tuple, list)):

        # single net
        torch.save(net.state_dict(), filename)
    else:

        # list of nets
        state_dict = [None] * len(net)
        for i in range(0, len(net)):
            state_dict[i] = net[i].state_dict()
        torch.save(state_dict, filename)


def load_net(net: tuple[nn.Module] | list[nn.Module] | nn.Module, filename: str) -> None:
    """Load a network from file.

        :param net: A pre-allocated Pytorch net (or list/tuple of nets), already moved to the right device.
        :param filename: The source file.
    """
    if not isinstance(net, (tuple, list)):

        # single net
        state_dict = torch.load(filename, map_location=net.device)
        net.load_state_dict(state_dict)
    else:

        # list of nets
        state_dict = torch.load(filename, map_location=net[0].device)
        for i in range(0, len(net)):
            net[i].load_state_dict(state_dict[i])
