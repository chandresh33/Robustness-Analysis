import pickle as pkl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

file_name_og = "Layer_surface_code\\cnn_mnist_models\\model-original-2.ckpt"
file_name_rob = "Layer_surface_code\\cnn_mnist_models\\model-robust_2.ckpt"


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def train(device, train_loader):
    model = NeuralNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Save the model checkpoint
        torch.save(model.state_dict(), 'Layer_surface_code/cnn_mnist_models/model-{}.ckpt'.format(epoch))

    return model


def test(device, test_loader, eps, file_name):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    criterion = nn.CrossEntropyLoss()

    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(file_name))

    correct = 0
    adv_correct = 0
    misclassified = 0
    total = 0
    noises = []
    y_preds = []
    y_preds_adv = []

    for images, labels in test_loader:
        images = Variable(images.to(device), requires_grad=True)
        labels = Variable(labels.to(device))

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Add perturbation
        grad = torch.sign(images.grad.data)
        imgs_adv = torch.clamp(images.data + (eps * grad), 0, 1)

        adv_outputs = model(Variable(imgs_adv))

        _, predicted = torch.max(outputs.data, 1)
        _, adv_preds = torch.max(adv_outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        adv_correct += (adv_preds == labels).sum().item()
        misclassified += (predicted != adv_preds).sum().item()

        noises.extend((images - imgs_adv).data.numpy())
        y_preds.extend(predicted.data.numpy())
        y_preds_adv.extend(adv_preds.data.numpy())

    print('Accuracy of the network w/o adversarial attack on the 10000 test images: {} %'.format(100 * correct / total))
    print('Accuracy of the network with adversarial attack on the 10000 test images: {} %'.format(
        100 * adv_correct / total))
    print('Number of misclassified examples (as compared to clean predictions): {}/{}'.format(misclassified, total))
    print("\n")

    return 100*(correct/total), 100*(adv_correct/total), (misclassified, total)


def main(flag):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if flag == "train":
        model = train(device, train_loader)
    elif flag == "test":
        pertub = np.linspace(0.001, 0.3, 10)
        no_attack_og = []
        attack_og = []
        no_attack_rob = []
        attack_rob = []

        for i in range(len(pertub)):
            og = test(device, test_loader, pertub[i], file_name=file_name_og)
            rob = test(device, test_loader, pertub[i], file_name=file_name_rob)

            no_attack_og.append(og[0])
            attack_og.append(og[1])

            no_attack_rob.append(rob[0])
            attack_rob.append(rob[1])

        plt.figure(1)
        plt.plot(pertub, attack_og, 'r*')
        plt.plot(pertub, attack_og, 'r', label='Original Model')

        # plt.plot(pertub, no_attack_og, 'g*')
        # plt.plot(pertub, no_attack_og, 'g', label='Original Model')

        plt.plot(pertub, attack_rob, 'b*')
        plt.plot(pertub, attack_rob, 'b', label='Robust Model')

        # plt.plot(pertub, no_attack_rob, 'y', label='Robust Model')

        plt.ylabel('Model Accuracy (%)')
        plt.xlabel('Perturbation magnitude')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main("test")
