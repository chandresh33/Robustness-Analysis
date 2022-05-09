# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X3pUR3F3GqC_a2z3CHG_Mx-ZFqM5Z7--
"""

# import os, sys
# from google.colab import drive
# drive.mount('/content/mnt')
# # nb_path = '/content/notebooks'
# # # os.symlink('/content/mnt/My Drive/Colab Notebooks', nb_path)
# # sys.path.insert(0, nb_path)  # or append(nb_path)
#
# !pip install --target=$nb_path torch~=1.7.0 torchvision pytorch-lightning
# !pip install test_tube

import torch
from torchvision.models import resnet50
from torch import nn
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.loggers import TestTubeLogger

from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)


class ResNetMNIST(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = resnet50(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    @auto_move_data
    def forward(self, xx):
        return self.model(xx)

    def training_step(self, batch_, batch_no):
        xx, yy = batch_
        logits = self(xx)
        loss = self.loss(logits, yy)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)


def train_test_loader():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset_ = torchvision.datasets.MNIST(root=r"MNIST_DATA\\", train=True,
                                                transform=trans, download=True)
    test_dataset_ = torchvision.datasets.MNIST(root=r"MNIST_DATA\\", train=False, transform=trans)

    train_loader_ = DataLoader(dataset=train_dataset_, batch_size=batch_size, shuffle=True)
    test_loader_ = DataLoader(dataset=test_dataset_, batch_size=1, shuffle=True)

    return train_loader_, test_loader_


def get_prediction(x_, model: pl.LightningModule):
    model.freeze()  # prepares model for predicting
    probabilities = torch.softmax(model(x_), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


def fgsm_attack(image_, epsilon_, data_grad_):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad_.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image_ = image_ + epsilon_*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image_ = torch.clamp(perturbed_image_, 0, 1)
    # Return the perturbed image
    return perturbed_image_


def trainer(train_loader_, model_):
    trainer_ = pl.Trainer(
        gpus=0,
        max_epochs=1,
        progress_bar_refresh_rate=20
    )

    trainer_.fit(model_, train_loader_)

    return trainer_


def get_accuracy(eval_model):
    total = 0
    correct = 0
    eval_model.eval()
    eval_model.freeze()
    with torch.no_grad():
        for images, labels in test_loader:
            output = eval_model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (correct / total)
        print('Test Accuracy of the original model on the 10000 test images: {}%'.format((correct / total) * 100))
        # res_array.append(accuracy)

        return accuracy


def get_model_performance(test_loader_, model):
    true_y, preds_y = [], []
    for batch in test_loader_:
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x, model)
        preds_y.extend(preds.cpu())

    print(classification_report(true_y, preds_y, digits=3))


def test_harness(model_, device_, test_loader_, epsilon_):

    y_, y_hat_, adv_examples = [], [], []
    correct = 0
    incorrect = 0
    final_acc = 0

    model_.eval()
    for data, target in test_loader_:
        y_.extend(target)
        data, target = data.to(device_), target.to(device_)

        data.requires_grad = True

        output, prob_ = get_prediction(data, model_)
        init_pred = output.max()

        if init_pred.item() != target.item():
            y_hat_.extend(output.cpu())
            continue

        prob_ = prob_.to(device_)
        loss = F.nll_loss(prob_, target)
        model_.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon_, data_grad)

        final_preds, _ = get_prediction(perturbed_data, model_)

        y_hat_.extend(final_preds.cpu())

        if final_preds.item() == target.item():
            correct += 1
        else:
            incorrect += 1

    final_acc = correct/float(len(test_loader))
    print("Epoch {}, Epsilon: {}\tTest Accuracy = {} / {} = {}".format(i*5, epsilon_, correct,
                                                                           len(test_loader), final_acc))

    return final_acc


num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
train_loader, test_loader = train_test_loader()

resnet_18 = ResNetMNIST()
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# trainer.save_checkpoint("resnet18_mnist.pt")
# inference_model = ResNetMNIST.load_from_checkpoint("resnet18_mnist.pt", map_location=device)
# resnet_18 = trainer.model
# epsilon = 1.0

# resnet_18 = ResNetMNIST()

epsilon = torch.arange(0, 1.1, 0.1)
model_performance = torch.zeros(5, len(epsilon))
# inference_model = ResNetMNIST
# trainer_temp = trainer(train_loader, inference_model)
# trainer_temp.save_checkpoint("ResNet/resnet18_mnist.pt")

for i in range(0, 5, 1):
    inference_model = ResNetMNIST.load_from_checkpoint("ResNet/Models/resnet50_mnist_version2.pt",
                                                       map_location=device)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        default_root_dir="ResNet"
    )

    trainer.fit(inference_model, train_dataloader=train_loader)
    trainer.save_checkpoint("ResNet/resnet50_mnist_version2.pt")
    resnet_18 = trainer.model

    for j in range(len(epsilon)):
        final_accuracy = test_harness(resnet_18, device, test_loader, epsilon[j])
        model_performance[i, j] = final_accuracy

print(model_performance)
torch.save(model_performance, "ResNet/resnet_18_20epochs.pt")
