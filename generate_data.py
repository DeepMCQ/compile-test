import numpy as np
from torch import nn
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
from sklearn.datasets import dump_svmlight_file
import torch
from tqdm import tqdm

def main_cifar():
    cifar = CIFAR10("cifar", train=True)
    downsample = PCA(384).fit(cifar.data.reshape(50000, -1))
    result = downsample.transform(cifar.data.reshape(50000, -1))
    result = torch.from_numpy(result)
    randPerm = torch.randperm(result.shape[-1])
    result = result[:, randPerm]
    result = torch.nn.functional.normalize(result)
    # [5, n, d]
    noise = torch.randn([5, *result.shape]) * (result.std(0) / 100)
    result = result + noise
    result = result.reshape(-1, result.shape[-1])
    result = torch.nn.functional.normalize(result)
    dump_svmlight_file(result.numpy(), y=list(range(len(result))), f="database", zero_based=False)
    query = torch.randperm(len(result))[:16384]
    dump_svmlight_file(result[query].numpy(), y=query.numpy(), f="query", zero_based=False)
    return len(result)

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]

@torch.no_grad()
def main_deep():
    transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(_IMG_MEAN, _IMG_STD, True)
        ]
    )
    cifar = CIFAR10("cifar", train=True, transform=transform)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, 384, bias=False)

    model = model.eval()#.cuda()

    dataLoader = DataLoader(cifar, 128, True, num_workers=8, pin_memory=True)

    allTensors = None

    try:
        while True:
            for x, y in tqdm(dataLoader, leave=False):
                # x = x.cuda(non_blocking=True)
                z = model(x).cpu()
                if allTensors is None:
                    allTensors = z
                else:
                    allTensors = torch.cat([allTensors, z])
                if len(allTensors) >= 231000:
                    raise StopIteration
    except StopIteration:
        pass
    result = torch.nn.functional.normalize(allTensors)
    dump_svmlight_file(result.numpy(), y=list(range(len(result))), f="database", zero_based=False)
    query = torch.randperm(len(result))[:16384]
    dump_svmlight_file(result[query].numpy(), y=query.numpy(), f="query", zero_based=False)
    return len(result)

def main():
    result = torch.randn([231400, 384])
    result = torch.nn.functional.normalize(result)
    dump_svmlight_file(result.numpy(), y=list(range(len(result))), f="database", zero_based=False)
    query = torch.randperm(len(result))[:16384]
    dump_svmlight_file(result[query].numpy(), y=query.numpy(), f="query", zero_based=False)
    return len(result)


if __name__ == "__main__":
    # print(main_cifar())
    print(main_deep())
