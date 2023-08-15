import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(True)

    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


@torch.no_grad()
def validation(model, loader, loss_fn):
    print("------------------Start validation------------------")
    losses = []
    model.eval()
    for image, target in tqdm(loader):
        image, target = image.to(device), target.float().to(device)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()
