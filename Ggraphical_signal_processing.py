import random
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
import copy

from collections import defaultdict
import networkx
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal as spy_sig

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import torchvision
from torchvision import transforms

import torch
from torch.utils.data import DataLoader

random_seed = 1
torch.manual_seed(random_seed)

device = "cpu"

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 5)


def load_dataset(download=False, dataset='cifar'):
    dataset = dataset.lower()
    batch_size = 100

    trans_og = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    AlexTransform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    imagenet_trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trans = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trans = trans_og

    if dataset == 'cifar':
        print("CIFAR10")
        train_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=True,
                                                      transform=trans, download=download)
        test_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=False, transform=trans)

    elif dataset == 'mnist':
        print("MNIST")
        train_dataset_ = torchvision.datasets.MNIST(root=r"MNIST_data/", train=True,
                                                    transform=trans, download=download)
        test_dataset_ = torchvision.datasets.MNIST(root=r"MNIST_data/", train=False, transform=trans)

    else:
        train_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10",
                                                      transform=trans, download=download)
        test_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=False, transform=trans)

    train_loader_ = DataLoader(dataset=train_dataset_, num_workers=2, batch_size=batch_size, shuffle=True)
    test_loader_ = DataLoader(dataset=test_dataset_, batch_size=1, shuffle=True)

    return train_loader_, test_loader_


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = torch.nn.CrossEntropyLoss()

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=learning_rate)


def return_same(x):
    return x


def test_model(model_, test_set_, fil):
    model_ = model_.model
    model_.eval()

    correct, total = 0, 0
    for data, target in test_set_:
        output = model_(data)

        if torch.argmax(output) == target.item():
            correct += 1
        total += 1

    model_acc = correct / total
    print("Model Accuracy: ", model_acc)

    return model_acc


def compare_kernels(x):
    # x = x.detach().numpy()
    temp_array_0 = []
    for i in range(len(x)):
        temp_array_1 = []
        for j in range(len(x)):
            diff_var = np.linalg.norm(x[i][0] - x[j][0], ord='fro')
            # x1 = x[i][0]
            # x2 = x[j+1][0]
            # c = cosine_similarity(x1, x2)

            temp_array_1.append(diff_var)

        temp_array_0.append(temp_array_1)

    temp_array_0 = np.array(temp_array_0)
    return temp_array_0


def similarity_filtering(adj_mat):

    def window_filter(matrix, alpha):
        window_min = alpha - (window_size/2)
        window_max = alpha + (window_size/2)

        matrix[matrix <= window_min] = 0
        matrix[matrix >= window_max] = 0
        return matrix

    temp_mat = copy.deepcopy(adj_mat)

    q_val = np.linspace(np.max(temp_mat[np.nonzero(temp_mat)])+0.2, np.min(temp_mat[np.nonzero(temp_mat)])-0.2, 100)
    con_mat = np.ones(len(q_val)*len(adj_mat)).reshape(len(q_val), len(adj_mat))

    index_mat = []
    window_size = 0.1
    connectivity_dict = {}

    for q in range(len(q_val)):
        temp_mat[temp_mat >= q_val[q]] = 0

        # temp_mat = copy.deepcopy(adj_mat)
        # temp_mat = window_filter(temp_mat, q_val[q])

        connectivity_dict[q_val[q]] = getRoots(graph_2_dict(temp_mat))

        for col in range(len(temp_mat)):
            nonzero_count = np.count_nonzero(temp_mat[col])
            temp_index = list(np.where(temp_mat[col] == 0)[0])

            for element in temp_index:
                if element not in index_mat:
                    index_mat.append(element)

            if nonzero_count < int(len(temp_mat[col])/2):
                con_mat[q, col] = 0

    return_tuple = (con_mat, q_val)
    return_tuple_2 = (index_mat, connectivity_dict)
    return return_tuple, return_tuple_2


def check_symetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def getRoots(aNeigh):

    def findRoot(aNode, aRoot):
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return aNode, aRoot[aNode][1]

    myRoot = {}
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode, 0)
    for myI in aNeigh:
        for myJ in aNeigh[myI]:
            (myRoot_myI, myDepthMyI) = findRoot(myI, myRoot)
            (myRoot_myJ, myDepthMyJ) = findRoot(myJ, myRoot)
            if myRoot_myI != myRoot_myJ:
                myMin = myRoot_myI
                myMax = myRoot_myJ
                if myDepthMyI > myDepthMyJ:
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (myMax, max(myRoot[myMin][1]+1, myRoot[myMax][1]))
                myRoot[myMin] = (myRoot[myMax][0], -1)
    myToRet = {}
    for myI in aNeigh:
        if myRoot[myI][0] == myI:
            myToRet[myI] = []
    for myI in aNeigh:
        myToRet[findRoot(myI, myRoot)[0]].append(myI)
    return myToRet


def graph_2_dict(adj_mat):
    return {i: [j for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(adj_mat)}


def sin_filter_func(input_mat, sin_params, sample_num):
    A_, B_, C_, D_ = sin_params
    samples_start, samples_end = np.min(input_mat), np.max(input_mat)

    x = np.linspace(samples_start, samples_end, sample_num)
    q = 0.05
    sin_filter = A_ * np.sin(((2*np.pi)/B_)*(x+C_))+D_
    bins = q * np.round(sin_filter / q)

    plt.plot(x, bins)
    plt.plot(x, sin_filter)
    plt.ylabel('Knockout Magnitude')
    plt.xlabel('Threshold Value')
    plt.show()

    for b in range(len(x)-1):
        ind = np.where(np.logical_and(input_mat <= x[b+1], input_mat >= x[b]))

        if input_mat[ind].size != 0:
            # print(input_mat[ind], bins[b])
            input_mat[ind] *= bins[b]

    return input_mat


learning_rate = 0.001
model = Model()
file_path = 'ResNet/MNIST_models/ResNet18_MNIST_40epochs.ckpt'
checkpoint = torch.load(file_path, map_location=device)
model.load_state_dict(checkpoint)
train_loader, test_loader = load_dataset(download=False, dataset='mnist')

# print(model.model.children)

weight = model.model.conv1.weight.detach().numpy()

rs = np.random.RandomState(42)
W = rs.uniform(size=(5, 5))
W[W < 0.3] = 0
W = W + W.T
W_dict = graph_2_dict(W)

np.fill_diagonal(W, 0)

similarity_array = compare_kernels(weight)
con_mat_full, index_connectivity = similarity_filtering(similarity_array)

ordered_indexes = index_connectivity[0]
components_dict = index_connectivity[1]

connectivity_vs_q_val = con_mat_full[0]
q_values = con_mat_full[1]

freq_range = np.arange(1, 16)
print(freq_range)
freq_results = []

for f in freq_range:
    freq = 1
    A = 0.5
    B = (np.max(weight) - np.min(weight))*(1/freq)
    C = 0.05
    D = 0.5
    sample_no = round(weight.size)

    params = (A, B, C, D)
    # test_model(model, test_loader, 1)  # 0.9777

    fin_mat = sin_filter_func(weight, params, sample_no)
    break
    print("here")

    model.model.layer3[0].conv2.weight = torch.nn.Parameter(torch.tensor(fin_mat))
    freq_results.append(test_model(model, test_loader, 1))  # 0.36, 0.4034

plt.plot(freq_range, freq_results)
plt.show()

# G = networkx.Graph(graph_2_dict(W))
# print(networkx.algebraic_connectivity(G))
# print(networkx.is_connected(G))

# for qq in connectivity_vs_q_val:
#     temp_plot.append(np.mean(qq))

# for q_dicts in components_dict.values():
#     temp_plot.append(len(q_dicts.keys()))

# plt.plot(q_values, temp_plot, color='darkgreen', label='ResNet18-MNIST (Trained for 40 epochs)')
# plt.xlabel('alpha')
# plt.ylabel('number of components')
# plt.title('Resulting Graph Components With Varying Threshold')
# plt.grid(alpha=0.5)
# plt.legend()
# # plt.savefig("plots/ResNet18-MNIST_40epochs_window_components.pdf")
# plt.show()

# similarity_array[similarity_array < np.mean(similarity_array)+0.3] = 0
#
# G = graphs.Graph(similarity_array)
# G.set_coordinates('ring2D')
# G.plot()
#
# plt.show()
