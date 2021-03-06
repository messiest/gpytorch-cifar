#!/usr/bin/env python
# Deep Kernel Learning with gpytorch
## Source: https://github.com/cornellius-gp/gpytorch/tree/master/examples/08_Deep_Kernel_Learning
import os
import math
import warnings
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import gpytorch
from tqdm import tqdm

from densenet import DenseNet

warnings.simplefilter('ignore')  # filter deprecation warnings


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('-epochs', type=int, default=500, help='total epochs')
parser.add_argument('-batch_size', type=int, default=64, help='batch size')
parser.add_argument('-num_workers', type=int, default=4, help='number of workers')
parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enable GPU acceleration')
args = parser.parse_args()

CUDA = args.cuda if torch.cuda.is_available() else False  # got GPU?


class DenseNetFeatureExtractor(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        return out


class GaussianProcessLayer(gpytorch.models.AdditiveGridInducingVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=128):
        super(GaussianProcessLayer, self).__init__(
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                num_dim=num_dim,
                mixing_params=False,
                sum_output=False,
            )
        self.cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                log_lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1),
                    math.exp(1),
                    sigma=0.1,
                    log_transform=True,
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res


def train(epoch, lr=0.1):
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalMarginalLogLikelihood(
        likelihood,
        model,
        num_data=len(train_loader.dataset)
    )
    desc = "Epoch: {:4d} | Loss: {: 4.4f}"
    pbar = tqdm(train_loader, desc=desc)
    for batch_idx, (data, target) in enumerate(pbar):
        if CUDA: data, target = data.cuda(), target.cuda()  # got CUDA?
        optimizer.zero_grad()
        output = model(data)
        loss = -1 * mll(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 50 == 0 or batch_idx == 0:
            pbar.set_description(desc.format(epoch, loss))

def test(epoch):
    model.eval()
    likelihood.eval()
    correct = 0
    for data, target in test_loader:
        if CUDA: data, target = data.cuda(), target.cuda()  # no CUDA!
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    results = f'Test:  {epoch:4d} | Accuracy: {100. * correct / len(test_loader.dataset):.2f}%'
    print(results)
    with open('logs.txt', 'a') as file:
        file.write(results + '\n' )


if __name__ == "__main__":
    dataset = args.dataset
    print(f"{dataset}".upper())

    # Data Augmentation
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    crop = transforms.RandomCrop(32, padding=4)
    flip = transforms.RandomHorizontalFlip()
    common_trans = [transforms.ToTensor(), normalize]

    train_compose = transforms.Compose([crop, flip] + common_trans)
    test_compose = transforms.Compose(common_trans)

    # Download Data
    batch_size = args.batch_size
    workers = args.num_workers
    if dataset == 'cifar10':
        d_func = datasets.CIFAR10
        train_set = datasets.CIFAR10('data', train=True, transform=train_compose, download=True)
        test_set = datasets.CIFAR10('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        num_classes = 10
    elif dataset == 'cifar100':
        d_func = datasets.CIFAR100
        train_set = datasets.CIFAR100('data', train=True, transform=train_compose, download=True)
        test_set = datasets.CIFAR100('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        num_classes = 100
    else:
        raise RuntimeError('dataset must be either "cifar10" or "cifar100"')

    feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
    num_features = feature_extractor.classifier.in_features

    model = DKLModel(feature_extractor, num_dim=num_features)

    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
        num_features=model.num_dim,
        n_classes=num_classes,
    )

    if CUDA:
        model = model.cuda()
        likelihood = likelihood.cuda()

    n_epochs = args.epochs
    lr = args.lr
    optimizer = SGD(
        [{'params': model.feature_extractor.parameters()},
         {'params': model.gp_layer.hyperparameters(), 'lr': 0.01 * lr},
         {'params': model.gp_layer.variational_parameters()},
         {'params': likelihood.parameters()}],
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=0,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[0.5 * n_epochs, 0.75 * n_epochs],
        gamma=0.1,
    )

    print("CUDA enabled:", CUDA)

    for epoch in range(1, n_epochs + 1):
        scheduler.step()
        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
            train(epoch)
            if epoch % 10 == 0:
                test(epoch)
                state_dict = model.state_dict()
                likelihood_state_dict = likelihood.state_dict()

                torch.save(
                    {
                        'model': state_dict,
                        'likelihood': likelihood_state_dict
                    },
                    f'checkpoints/dkl_{dataset}_checkpoint_{epoch}.dat',
                )
