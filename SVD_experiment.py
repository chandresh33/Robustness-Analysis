import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import linalg
from scipy.integrate import simps

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import copy
import surface3d_demo as sd3

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# pretrained_model_path = 'Layer_surface_code\\cnn_mnist_models\\model-4.ckpt'
pretrained_model_path = 'Layer_surface_code\\cnn_mnist_models\\cnn_mnist_model_2.ckpt'

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset_ = torchvision.datasets.MNIST(root=r"Layer_surface_code\MNIST_DATA\\", train=False, transform=trans)
test_loader = DataLoader(dataset=test_dataset_, batch_size=batch_size, shuffle=True)

file_name_1 = "Layer1_robustness\Layer1_data\layer1_data\layer1_model2.txt"
file_name_2 = "Layer2_robustness\Layer2_data\Layer2_data\layer2_model2.txt"


def get_xyz(file_name):
    x_, y_, z_ = sd3.get_data(file_name)
    return x_, y_, z_


def layer_rob_score(des_acc, f_name):
    x, y, z = get_xyz(f_name)
    return sd3.integral_simps(x, y, z, des_acc)


# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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

    def forward(self, param):
        out = self.layer1(param)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def get_accuracy(eval_model):
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            output = eval_model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        # print('Test Accuracy of the original model on the 10000 test images: {}%'.format((correct / total) * 100))
        # res_array.append(accuracy)

        return accuracy


def unfolding(n_, A_):
    shape = A_.shape
    size = np.prod(shape)
    lsize = size // shape[n_]
    sizelist = list(range(len(shape)))
    sizelist[n_] = 0
    sizelist[0] = n_

    return A_.permute(sizelist).reshape(shape[n_], lsize)


def modelsvd(n_, A_):
    nA = unfolding(n_, A_)
    return torch.svd(nA)


def hosvd(A):
    ulist = []
    S = A
    for i, ni in enumerate(A.shape):
        u, _, _ = modelsvd(i, A)
        ulist.append(u)
        S = torch.tensordot(S, u.t(), dims=([0], [0]))

        return S, ulist


def matrix_difference(matrix_a, matrix_b):
    row1, col1 = len(matrix_a), len(matrix_a[0])
    row2, col2 = len(matrix_b), len(matrix_b[0])

    flag = True

    if row1 != row2 or col1 != col2:
        print("Matrices are not equal")

    else:
        for i in range(0, row1):
            for j in range(0, col1):
                if matrix_b[i][j] != matrix_b[i][j]:
                    flag = False
                    break

        if flag:
            print("Matrices are equal")
        else:
            print("Matrices are not equal")


def d2_different(mat_a, mat_b):
    return torch.sqrt(torch.sum(torch.pow(mat_a.sub(mat_b), 2)))


def get_data(file):
    accuracy_vals = np.loadtxt(file, dtype=float)

    x_ = np.arange(len(accuracy_vals))
    y_ = np.arange(len(accuracy_vals[0]))

    y_mesh, x_mesh = np.meshgrid(y_, x_)

    return x_mesh, y_mesh, accuracy_vals


def r_truncation(matrix):
    beta = matrix.size()[1]/matrix.size()[0]
    omega = 0.56*pow(beta, 3) - 0.95*pow(beta, 2) + 1.82*beta + 1.43
    u_, d_, v_ = torch.svd(matrix)
    y_ = torch.diag(matrix)
    y_[y_ < omega*torch.median(y_)] = 0
    xhat = torch.matmul(torch.matmul(u_, torch.diag_embed(y_)), v_.t())
    return xhat


def find_cutoff(hk, ak):
    lip_constant = 0.5
    kk_2 = []
    for i in range(len(ak)):
        for j in range(len(hk)):
            kk_2.append((abs(1+(hk[j]*ak[i]))-1)/(abs(hk[j])*lip_constant))
    return min(kk_2)


def rob_score_varied(min_acc, no_of_samples):
    acc = 99.15
    laye = layer_rob_score(acc, file_name_1)
    arr_max = np.max(layer_rob_score(acc, file_name_1)[2])
    arr_min = min_acc
    range_array = np.linspace(arr_max, arr_min, no_of_samples)
    empty_array = []

    for i in range_array:
        empty_array.append(abs(layer_rob_score(i, file_name_1)[1]))

    empty_array = np.array(empty_array)

    # plt.plot(range_array, np.log(empty_array))
    # plt.xlabel('log(A)')
    # plt.ylabel('desired accuracy')
    # plt.show()

    return empty_array


def hk_robustness(u, s, vh, model_, matrix):
    ak_vec = np.arange(0.05, 1, 0.05)
    hk_vec = np.arange(0.1, 3, 0.1)
    accuracy = np.zeros((len(ak_vec), len(hk_vec)))

    for i in range(len(ak_vec)):
        for j in range(len(hk_vec)):
            cutoff = find_cutoff(ak_vec, hk_vec)
            print(cutoff)
            r = np.max(np.where(s > cutoff))

            xclean = u[:, :(r + 1)] @ np.diag(s[:(r + 1)]) @ vh[:(r + 1), :]
            xclean = torch.from_numpy(xclean)

            final_weights = xclean.reshape(matrix.size())
            model_.layer1[0].weight = nn.Parameter(final_weights)
            accuracy[i][j] = get_accuracy(model_)
            print(get_accuracy(model_))

    return accuracy


def get_st_plot(matrix, file_name):
    des_accs_ = np.linspace(99.25, 1, 100)
    tresh_vals_ = []

    u_, s_, vh_ = np.linalg.svd(matrix, full_matrices=True)
    N_, M_ = u_.shape[0], vh_.shape[0]

    for i in range(len(des_accs_)):
        robust_obj_1 = layer_rob_score(des_accs_[i], file_name)
        noise_mag_1 = (robust_obj_1[0] - robust_obj_1[1]) / (robust_obj_1[2] - robust_obj_1[1])

        ans_vals = op_hard_thresholding(M_, N_, 1, sigma=np.std(matrix) * noise_mag_1)
        tresh_vals_.append(ans_vals)

    tresh_vals_ = ((tresh_vals_ - np.min(tresh_vals_)) / (np.max(tresh_vals_)) * np.min(s_)) + np.min(s_)
    des_accs_ = ((des_accs_ - np.min(des_accs_)) / (np.max(des_accs_) - np.min(des_accs_))) * (len(s_) - 1)

    return tresh_vals_, des_accs_, s_


def op_hard_thresholding(m, n, scaling,  sigma):
    beta = m/n
    lambda_1 = np.sqrt((2*(beta+1))+((8*beta)/((beta+1)+np.sqrt((beta**2)+(14*beta)+1))))
    lambda_2 = (4/np.sqrt(3)) * np.sqrt(n) * sigma
    t_str = (lambda_1*np.sqrt(n)*sigma)  # *scaling
    return t_str


def test_harness(weight_matrix, noise, ml_model, orig_weights):
    u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
    M, N = weight_matrix.shape[1], weight_matrix.shape[0]

    for i in s:
        print(i, ",")

    scaling = np.linspace(0.1, 0.8, 50)
    scaling_acc = []

    for i in range(len(s)):
        # cutoff = op_hard_thresholding(M, N,  scl_factor[i], sigma=noise)
        cutoff = s[len(s)-1-i]

        try:
            r = np.max(np.where(s > cutoff))
        except ValueError:
            r = np.max(np.where(s > np.min(s)))

        xclean = u[:, :(r + 1)] @ np.diag(s[:(r + 1)]) @ vh[:(r + 1), :]
        xclean = torch.from_numpy(xclean)

        # weight_matrix = torch.from_numpy(weight_matrix)

        final_weights = xclean.reshape(orig_weights.shape)
        ml_model.layer2[0].weight = nn.Parameter(final_weights)
        print(get_accuracy(ml_model), ",")

        scaling_acc.append(get_accuracy(ml_model))

    return scaling_acc


def t_value(m, n, y_med):
    beta = m/n
    wb = (0.56*(beta**3)) - (0.95*(beta**2)) + (1.82*beta) + 1.43
    return wb*y_med


def final_harness1(matrix, noise, ml_model, orig_weights):
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    N, M = u.shape[0], vh.shape[0]

    # cutoff = t_value(M, N, np.median(s))  # *scl_factor*noise_mag
    # cutoff = op_hard_thresholding(M, N, 1, sigma=noise)
    cutoff = noise

    print("cutoff 1: ", cutoff)
    print("s1: ", s)
    r = np.max(np.where(s > cutoff))

    xclean = u[:, :(r + 1)] @ np.diag(s[:(r + 1)]) @ vh[:(r + 1), :]

    # weight_matrix = torch.from_numpy(weight_matrix)

    final_weights = torch.from_numpy(xclean).reshape(orig_weights.shape)
    ml_model.layer1[0].weight = nn.Parameter(final_weights)
    print("Model accuracy: ", get_accuracy(ml_model))
    print("\n")

    return ml_model, xclean


def final_harness2(matrix, noise, ml_model, orig_weights):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    M, N = len(u[:, 0]), len(vh[0, :])

    y_median = np.median(s)

    # cutoff = t_value(M, N, y_median)  # *scl_factor*noise_mag
    # cutoff = op_hard_thresholding(M, N, 1, sigma=np.std(s)*noise)
    cutoff = noise

    print("cutoff 2: ", cutoff)
    print("s2: ", s)
    r = np.max(np.where(s > cutoff))

    xclean = u[:, :(r + 1)] @ np.diag(s[:(r + 1)]) @ vh[:(r + 1), :]

    # weight_matrix = torch.from_numpy(weight_matrix)

    final_weights = torch.from_numpy(xclean).reshape(orig_weights.shape)
    ml_model.layer2[0].weight = nn.Parameter(final_weights)
    print("Model accuracy: ", get_accuracy(ml_model))
    print("\n")

    return ml_model, xclean


# # weight layer
def load_test_weights1(model_path):

    try:
        model_ = ConvNet()
        model_.load_state_dict(torch.load(model_path))
    except AttributeError:
        model_ = torch.load(model_path)

    weights_ = model_.layer1[0].weight.clone().detach()

    flatten_kernel = weights_.size()[1]*weights_.size()[2]*weights_.size()[3]
    test_weights = weights_.reshape((weights_.size()[0], flatten_kernel))
    test_weights = np.array(test_weights)

    return test_weights, weights_, model_, np.linspace(0.01, 0.08, 50)


def load_test_weights2(model_path):

    try:
        model_ = ConvNet()
        model_.load_state_dict(torch.load(model_path))
    except AttributeError:
        model_ = torch.load(model_path)

    weights_ = model_.layer2[0].weight.clone().detach()

    flatten_kernel = weights_.size()[2]*weights_.size()[3]
    test_weights = weights_.reshape((weights_.size()[0]*weights_.size()[1], flatten_kernel))
    test_weights = np.array(test_weights)

    return test_weights, weights_, model_, np.linspace(0.01, 0.08, 50)


def test_desired_accuracy():
    holder = get_xyz(file_name_1)
    desired_accuracies = np.arange(np.max(holder[2]), 75, -0.1)
    recorded_noise_mag = []
    for i in range(len(desired_accuracies)):
        rob_obj = layer_rob_score(desired_accuracies[i], file_name_1)
        noise_mag_ = abs(rob_obj[0] - rob_obj[1]) / (len(holder[0][0, :]) * len(holder[1][:, 0]))
        recorded_noise_mag.append(noise_mag_)

    return np.array(recorded_noise_mag), desired_accuracies


test_w, original_weights, model, _ = load_test_weights1(pretrained_model_path)
print("Model accuracy: ", get_accuracy(model))
print("\n")

x, y, z = get_data(file_name_1)
z_rms_value = np.sqrt(np.mean(z**2))
z_rms = np.ones_like(z)*z_rms_value

z_over, z_under = copy.deepcopy(z), copy.deepcopy(z)
z_over[z_over <= z_rms] = z_rms_value
z_under[z_under >= z_rms] = z_rms_value

mid_model, _ = final_harness2(test_w, 1.380, model, original_weights)

test_w_, original_weights_, model_, _ = load_test_weights2(pretrained_model_path)
final_model, _ = final_harness2(test_w_, 1.380, model_, original_weights_)

torch.save(model.state_dict(), 'Layer_surface_code/cnn_mnist_models/model-robust_2.ckpt')
# torch.save(model.state_dict(), 'Layer_surface_code/cnn_mnist_models/model-original_2.ckpt')

rms_top = np.sqrt((np.square(z_over-z_rms_value)).mean(axis=None))
rms_bottom = np.sqrt((np.square(z_rms_value-z_under)).mean(axis=None))
snr = 20*np.log10(rms_top/rms_bottom)
print("snr: ", snr)

# u, s, vh = np.linalg.svd(test_w, full_matrices=False)
#
# mu, sigma = np.mean(test_w), np.std(test_w)*noise_mag_1
# noise = np.random.normal(mu, sigma, test_w.shape)
# u_, s_, vh_ = np.linalg.svd(noise, full_matrices=False)
# print("max noise s: ", np.max(s_), "max s: ", np.max(s))
#
# plt.plot(np.linspace(0, 1, len(noise[0, :])), s_, 'b*')
# plt.plot(np.linspace(0, 1, len(test_w[0, :])), s, 'ro')
# plt.show()
