import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
from scipy import signal
import copy
import surface3d_demo as sd3
# import seaborn as sns; sns.set()

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.manual_seed(0)
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
pretrained_model_path = 'Layer_surface_code/cnn_mnist_models/model-original_2.ckpt'

file_name_1 = "Layer1_robustness\\Layer1_average.txt"
file_name_2 = "Layer2_robustness\\Layer2_average.txt"


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
        # noise = torch.zeros_like(param).normal_(mean=0, std=0.1)
        # param += noise
        out = self.layer1(param)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class AddGaussNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_accuracy(eval_model):
    total = 0
    correct = 0
    eval_model.eval()
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


def singular_values_plot(original_weights_, s_weights, sbar_weights):
    s = np.linalg.svd(original_weights_, full_matrices=True)[1]
    s_s = np.linalg.svd(s_weights, full_matrices=True)[1]
    s_sbar = np.linalg.svd(sbar_weights, full_matrices=True)[1]

    plt.figure(1)
    plt.subplot(311)
    plt.plot(np.arange(len(s)), s, 'b*')
    plt.xticks(np.arange(len(s)))
    plt.ylabel("$\sigma (i)$")
    plt.xlabel('$M$')
    plt.title("Full Weight Matrix")

    plt.subplot(312)
    plt.plot(np.arange(len(s_s)), s_s, 'g*')
    plt.xticks(np.arange(len(s_s)))
    plt.ylabel("$\sigma (i)$")
    plt.xlabel('$M$')

    plt.subplot(313)
    plt.plot(np.arange(len(s_sbar)), s_sbar, 'r*')
    plt.xticks(np.arange(len(s_sbar)))
    plt.ylabel("$\sigma (i)$")
    plt.xlabel('$M$')

    plt.tight_layout()
    plt.show()


def integral_mse(xx, yy, zz, zz_under, zz_over):
    integral_diff = sd3.integral_simps(xx, yy, zz, zz_under)
    mse2 = abs(integral_diff[0] - integral_diff[1])
    integral_diff = sd3.integral_simps(xx, yy, zz, zz_over)
    mse2 = abs(integral_diff[1] - integral_diff[0]) / mse2
    print("mse 2: ", mse2)


def guassian_sigma(weight_mat, mat_size):
    acc_values = []
    scaling_values = np.linspace(0.01, 0.085, 20)

    for i in range(len(scaling_values)):
        mu_, sigma_ = np.mean(weight_mat), scaling_values[i]
        s_ = np.random.normal(mu_, sigma_, weight_mat.shape)

        u_weights, s_weights, vh_weights = np.linalg.svd(weight_mat, full_matrices=False)

        r = np.max(np.where(s_weights > np.max(s_)))
        xclean = u_weights[:, :(r + 1)] @ np.diag(s_weights[:(r + 1)]) @ vh_weights[:(r + 1), :]
        xclean_ = torch.from_numpy(xclean)

        final_weights_ = xclean_.reshape(mat_size)
        model.layer1[0].weight = nn.Parameter(final_weights_)
        print("model accuracy: ", get_accuracy(model))

        acc_values.append(get_accuracy(model))

    plt.plot(acc_values)
    plt.show()

    return acc_values


def load_test_weights(model_path, layer_):

    try:
        model_ = ConvNet()
        model_.load_state_dict(torch.load(model_path))
    except AttributeError:
        model_ = torch.load(model_path)

    if layer_ == 1:
        weights_ = model_.layer1[0].weight.clone().detach()
    elif layer_ == 2:
        weights_ = model_.layer2[0].weight.clone().detach()
    else:
        weights_ = model_.layer1[0].weight.clone().detach()

    flatten_kernel = weights_.size()[1]*weights_.size()[2]*weights_.size()[3]
    test_weights_ = weights_.reshape((weights_.size()[0], flatten_kernel))
    test_weights_ = np.array(test_weights_)

    return test_weights_, weights_, model_, np.linspace(0.01, 0.08, 50)


def get_data(file):
    accuracy_vals = np.loadtxt(file, dtype=float)

    x_values = np.arange(len(accuracy_vals))
    y_values = np.arange(len(accuracy_vals[0]))

    y_mesh, x_mesh = np.meshgrid(y_values, x_values)

    return x_mesh, y_mesh, accuracy_vals


def plot_rms(x_array, y_array, z_array, z_array_rms):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_array, y_array, z_array, cmap='Spectral_r', rstride=1,
                    cstride=1, alpha=0.75, antialiased=True)
    ax.plot_surface(x_array, y_array, z_array_rms, cmap='bone', rstride=1, cstride=1,
                    alpha=0.9, antialiased=True)

    plt.title("Surface Plot")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_zlabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()


def plot_colour_bar(z_original, z_greater, z_less):
    plt.subplot(131)
    orig = plt.imshow(z_original, cmap='jet_r', aspect='auto')
    plt.xlabel("Probability")
    plt.ylabel("Layers")
    plt.title('Original Surface plot')
    cbar = plt.colorbar(orig)
    cbar.set_label('Accuracy (%)')
    plt.subplot(132)
    above_rms = plt.imshow(z_greater, cmap='jet_r', aspect='auto', label='Above RMS')
    plt.xlabel("Probability")
    plt.ylabel("Layers")
    plt.title('Above RMS')
    cbar = plt.colorbar(above_rms)
    cbar.set_label('Accuracy (%)')
    plt.subplot(133)
    below_rms = plt.imshow(z_less, cmap='jet_r', aspect='auto', label='Below RMS')
    plt.xlabel("Probability")
    plt.ylabel("Layers")
    plt.title('Below RMS')
    cbar = plt.colorbar(below_rms)
    cbar.set_label('Accuracy (%)')

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def rms(value):
    return np.sqrt((np.square(value)).mean(axis=None))


def op_hard_thresholding(m, n, scaling,  sigma):
    beta = m/n
    print("beta: ", beta)
    lambda_1 = np.sqrt((2*(beta+1))+((8*beta)/((beta+1)+np.sqrt((beta**2)+(14*beta)+1))))
    lambda_2 = (4/np.sqrt(3)) * np.sqrt(n) * sigma
    t_str = (lambda_1*np.sqrt(n)*scaling)  # *scaling
    return t_str


def get_over_under(z_array):
    average = []
    for prob in z_array:
        average.append(np.sqrt(np.mean(prob**2)))

    average = np.array(average)

    average_over, average_under = copy.deepcopy(average), copy.deepcopy(average)
    average_rms = np.sqrt(np.mean(z_array**2))-0.02

    average_over[average_over >= average_rms] = 0
    average_under[average_under <= average_rms] = 0

    return np.array(average_over), np.array(average_under), average_rms


def trunc_weights(full_surface, old_weights):

    def find_indexes(feature_type, full_sur):
        return np.argwhere(feature_type == 0)

    surface_ = get_over_under(full_surface)

    index_under = find_indexes(surface_[0], surface_[2])
    index_under = index_under.reshape(len(index_under))

    index_over = find_indexes(surface_[1], surface_[2])
    index_over = index_over.reshape(len(index_over))

    over, under = copy.deepcopy(old_weights), copy.deepcopy(old_weights)

    over[index_under, :] = 0
    under[index_over, :] = 0

    return (over, index_over), (under, index_under)


def denoise_mat(matrix, noise, limit, pos_0, pos_1):
    u_, s_, vh_ = np.linalg.svd(matrix, full_matrices=True)
    M, N = u_.shape[0], vh_.shape[0]
    print("sigma: ", s_)

    if pos_1 == 1:
        # cutoff = t_value(M, N, np.median(s))  # *scl_factor*noise_mag
        cutoff = op_hard_thresholding(M, N, 0.03, sigma=noise)
    else:
        cutoff = noise

    print("threshold value:", cutoff, "\n")
    r = np.max(np.where(s_ > cutoff))

    xclean = u_[:, :(r + 1)] @ np.diag(s_[:(r + 1)]) @ vh_[:(r + 1), :]

    if pos_0 == 1:
        final_weights_ = torch.from_numpy(xclean)
    else:
        final_weights_ = torch.from_numpy(matrix)
        final_weights_[limit] = 0

    # final_weights = torch.from_numpy(xclean).reshape(orig_weights.shape)
    # ml_model.layer1[0].weight = nn.Parameter(final_weights)
    # print("Model accuracy: ", get_accuracy(ml_model))
    # print("\n")

    return final_weights_, xclean


def reconstruct_og_weights(og_weights_, over_torch_, over_index_, under_torch_, under_index_):
    og_weights_ = torch.tensor(og_weights_)
    mask_og = torch.zeros_like(og_weights_)

    for i in range(len(over_index_)):
        mask_og[over_index_[i]] = over_torch_[i]

    for j in range(len(under_index_)):
        mask_og[under_index_[j]] = under_torch_[j]*50

    return mask_og


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                            AddGaussNoise(0., 1.0*0.5)])
test_dataset_ = torchvision.datasets.MNIST(root=r"Layer_surface_code\MNIST_DATA\\", train=False, transform=trans)
test_loader = DataLoader(dataset=test_dataset_, batch_size=batch_size, shuffle=True)

layer = 2
weights, original_weights, model, _ = load_test_weights(pretrained_model_path, layer_=layer)
print("Model accuracy: ", get_accuracy(model))
print("\n")

# get the surface plot
if layer == 1:
    x, y, z = get_data(file_name_1)
else:
    x, y, z = get_data(file_name_2)

z_rms_value = np.sqrt(np.mean(z**2))
z_rms = np.ones_like(z)*z_rms_value

# make all values above rms equal rms and equally for below
z_over, z_under = copy.deepcopy(z), copy.deepcopy(z)
z_over[z_over <= z_rms] = z_rms_value
z_under[z_under >= z_rms] = z_rms_value
# plot_colour_bar(z, z_over, z_under)

# truncate the weights according to the surface plots
weights_s, weights_sbar = trunc_weights(z, weights)
over_index, under_index = weights_s[1], weights_sbar[1]
weights_s, weights_sbar = weights_s[0], weights_sbar[0]
print("over index: ", over_index)
print("under index: ", under_index)

# delete any 0's in the weights (previously set to the rms value)
weights_sbar = np.delete(weights_sbar, over_index, axis=0)
weights_s = np.delete(weights_s, under_index, axis=0)

# denoise the now truncated matracies
over_torch, over_np = denoise_mat(weights_s, 1, over_index, pos_0=1, pos_1=1)
under_torch, under_np = denoise_mat(weights_sbar, 0.0, under_index, pos_0=1, pos_1=0)

# complie the final weights as a combination of the denoised matracies
final_weights = reconstruct_og_weights(weights, over_torch, over_index, under_torch, under_index)

final_weights = final_weights.reshape(original_weights.shape)

if layer == 1:
    model.layer1[0].weight = nn.Parameter(final_weights)
else:
    model.layer2[0].weight = nn.Parameter(final_weights)

print("Robust model accuracy: ", get_accuracy(model))
print("\n")

torch.save(model.state_dict(), 'Layer_surface_code/cnn_mnist_models/model-robust_2.ckpt')

#####################################################################

# plot_rms(x, y, z_over, z_under)
# plot_colour_bar(z, z_over, z_under)

# plt.subplot(311)
# plt.plot(np.arange(len(s)), s, 'g*', label='Full matrix')
# plt.xticks(range(0, len(s)))
# plt.legend()
# plt.ylabel('$\sigma_i$')
# plt.xlabel('i, i = {1,..,M)')
#
# plt.subplot(312)
# plt.plot(range(len(s_s)), s_s, 'b*', label='Important features')
# plt.xticks(range(0, len(s_s)))
# plt.legend()
# plt.ylabel('$\sigma_i$')
# plt.xlabel('i, i = {1,...,t}, t<M')
#
# plt.subplot(313)
# plt.plot(np.arange(len(s_sbar)), s_sbar, 'r*', label='Unimportant features')
# plt.xticks(range(0, len(s_sbar)))
# plt.legend()
# plt.ylabel('$\sigma_i$')
# plt.xlabel('i, i = {1,...,M-t}, t<M')
#
# plt.tight_layout(pad=0.5)
# plt.show()

