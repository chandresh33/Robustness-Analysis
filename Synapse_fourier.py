import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


def load_dataset(download=False, dataset='cifar'):
  dataset = dataset.lower()
  batch_size = 100

  trans_og = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  # trans_og = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

  imagenet_trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # trans =  transforms.Compose(
  #   [transforms.ToTensor(),
  #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # trans = AlexTransform

  if dataset == 'cifar':
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print("CIFAR10")
    train_dataset_ = torchvision.datasets.CIFAR10(root=r"data/CIFAR10", train=True,
                                                  transform=trans, download=download)
    test_dataset_ = torchvision.datasets.CIFAR10(root=r"data/CIFAR10", train=False, transform=trans)

  elif dataset == 'mnist':
    trans_og = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    print("MNIST")
    train_dataset_ = torchvision.datasets.MNIST(root=r"data/", train=True,
                                                transform=trans_og, download=download)
    test_dataset_ = torchvision.datasets.MNIST(root=r"data/", train=False, transform=trans_og)

  elif dataset == 'imagenet':
    print("ImageNet")

    train_dataset = torchvision.datasets.ImageNet(root=r"Dataset/", split='train', download=download)
    test_dataset_ = torchvision.datasets.ImageNet(root=r"Dataset/", split='test', target_transform=trans)

  else:
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", image_set='test',
                                                  transform=trans, download=download)
    test_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=False, transform=trans)

  train_loader_ = DataLoader(dataset=train_dataset_, num_workers=2, batch_size=batch_size, shuffle=True)
  test_loader_ = DataLoader(dataset=test_dataset_, batch_size=1, shuffle=True)

  return train_loader_, test_loader_


class Model(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = torchvision.models.resnet18(num_classes=10)

    self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  ## googlenet
    # self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

    self.loss = nn.CrossEntropyLoss()

  @auto_move_data
  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_no):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.model.parameters(), lr=0.001)


def model_tester(model_, test_set_):
  model_.eval()
  model_.to(device)
  criterion = nn.CrossEntropyLoss()

  # para_loader = pll.ParallelLoader(test_set_, [device]).per_device_loader(device)

  correct, total, running_loss = 0, 0, 0
  with torch.no_grad():
    for images, labels in test_set_:
      data = images.to(device)
      target = labels.to(device)

      output = model_(data)

      _, predicted = torch.max(output.data, 1)
      test_loss = criterion(output, target).to(device)

      total += target.size(0)
      correct += (predicted == target).sum().item()
      running_loss += test_loss.item() * data.size(0)

  model_acc = correct / total
  epoch_loss = running_loss / len(test_set_)

  return model_acc, epoch_loss


def _rebuild_xla_tensor(data, dtype, device, requires_grad):
  tensor = torch.from_numpy(data).to(dtype=dtype, device='cpu')
  tensor.requires_grad = requires_grad
  return tensor


def step_filter_func(input_mat, offset):
  y = torch.zeros_like(input_mat)
  with torch.no_grad():
    input_mat_fin = torch.where((input_mat >= offset), y, input_mat)
    # input_mat[ind] *= 0

  return input_mat_fin


def fgsm_attack(image, epsilon, data_grad):
  # Collect the element-wise sign of the data gradient
  sign_data_grad = data_grad.sign()
  # Create the perturbed image by adjusting each pixel of the input image
  perturbed_image = image + epsilon * sign_data_grad
  # Adding clipping to maintain [0,1] range
  # perturbed_image = torch.clamp(perturbed_image, 0, 1)

  # Return the perturbed image
  return perturbed_image


def fgsm_test(model, device, test_loader, epsilon):

  # Accuracy counter
  correct = 0
  adv_examples = []
  model.eval()

  # Loop over all examples in test set
  for data, target in test_loader:

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    # If the initial prediction is wrong, dont bother attacking, just move on
    # if init_pred.item() != target.item():
    #   continue

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    if final_pred.item() == target.item():
      correct += 1
      # Special case for saving 0 epsilon examples
      if (epsilon == 0) and (len(adv_examples) < 5):
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    else:
      # Save some adv examples for visualization later
      if len(adv_examples) < 5:
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

  # Calculate final accuracy for this epsilon
  final_acc = correct / float(len(test_loader))
  print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

  # Return the accuracy and an adversarial example
  return final_acc, adv_examples


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch._utils._rebuild_xla_tensor = _rebuild_xla_tensor

file_name = 'Saved_models/ResNet18_MNIST_100epochs.ckpt'
checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

model = Model()
model.load_state_dict(checkpoint)
train_loader, test_loader = load_dataset(download=False, dataset='mnist')
#
# print("OG Model Accuracy: ", model_tester(model, test_loader))  # Accuracy = 0.9923 (MNIST)
# print(model.model.conv1.weight.max()) # 0.7687

weight = model.model.conv1.weight.detach()
samples_no = 25
steps = ((torch.max(weight) - torch.min(weight)) / samples_no) / 2
alpha_vals = torch.linspace(torch.max(weight), torch.min(weight), samples_no, device=device)
# print(weight[0][0])
# print("\n")

# for i in tqdm.tqdm(alpha_vals):
#   filtered_weight = step_filter_func(weight, i)
#   print(filtered_weight[0][0])
#   print("\n")
  #model.model.conv1.weight = nn.Parameter(filtered_weight)
  #print("Perturbed Model Accuracy: ", model_tester(model, test_loader))  # Accuracy = 0.1135 (MNIST)

weight = model.model.conv1.weight.detach()
w_shape = weight.shape
weight = weight.reshape(w_shape[1], w_shape[0], w_shape[2]*w_shape[3])

fft_w = torch.fft.fftn(weight[0])
fft_w = torch.fft.fftshift(fft_w)

filter = torch.ones_like(fft_w.real)
filter[0:1, :] = 0
filter[49:50, :] = 0

filtered_weight = fft_w*filter
filtered_weight = torch.fft.ifftn(filtered_weight)

# acc, examples = fgsm_test(model.model, device, test_loader, 0.45)
#
# print(examples.shape)

# plt.imshow(filtered_weight.real, cmap='Greys')
# plt.title("Filtered Weights")
# plt.show()

# plt.figure(figsize=(10,20))
# plt.subplot(211)
# plt.imshow(fft_w.real.numpy(), cmap='Greys')
# plt.title("Fourier Space")
#
# # plt.subplot(312)
# # plt.imshow(fft_w.imag.numpy(), cmap='Greys')
#
# plt.subplot(212)
# plt.imshow(weight[0].numpy(), cmap='Greys')
# plt.title("Layer Weights")
#
# # plt.legend()
# plt.show()
