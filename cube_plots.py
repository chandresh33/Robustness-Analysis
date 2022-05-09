import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from itertools import product
import pickle
import torch
import scipy.integrate as integrate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _rebuild_xla_tensor(data, dtype, device, requires_grad):
  tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
  tensor.requires_grad = requires_grad
  return tensor

torch._utils._rebuild_xla_tensor = _rebuild_xla_tensor

def load_pickle(filename):
  objects = []
  with (open(filename+".pkl", "rb")) as openfile:
    while True:
      try:
        objects.append(pickle.load(openfile))
      except EOFError:
        break
  return objects


def load_cube_data(dir_apth_, model_type_, dataset_name_, analysis_type_, type_filter_):
    epochs = np.arange(10, 101, 10)
    epoch_dict = {}

    for e in epochs:
      path = dir_apth_ + model_type_ + '/'
      file_name = path+dataset_name_+'/'+analysis_type_+'/'+model_type_+'_comb2_'+type_filter_+'_' + dataset_name + '_' + str(e) + 'epochs'
      # file_name = path+dataset_name_+'/'+analysis_type_+'/'+model_type_+'_'+type_filter_+'_advog_' + dataset_name + '_' + str(e) + 'epochs'

      data = load_pickle(file_name)[0]
      epoch_dict[e] = data

    # print(epoch_dict[10][0.0][-1])  # epochs -> epsilons -> alpha

    epoch_array = []
    for ep in epoch_dict:
        ep_data = epoch_dict[ep]
        epsilon_array = []

        for epsi in ep_data:
            epsi_data = ep_data[epsi]
            alpha_array = []

            for alpha in epsi_data:
                alpha_data = epsi_data[alpha]
                acc_adv = alpha_data[1]
                alpha_array.append(acc_adv)

            epsilon_array.append(alpha_array)
        epoch_array.append(epsilon_array)

    epoch_array = np.array(epoch_array)
    return epoch_array


def eigen_compress(mat):
    r = 1
    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    recon_mat = u[:1, :r] @ np.diag(s[:r]) @ vh[:r, :]
    return recon_mat


def combine_response(mat_a, mat_b, mat_c):
    mat = zip(mat_a, mat_b, mat_c)
    res_mat = []
    for epoch in mat:
        mat2 = zip(epoch[0], epoch[1], epoch[2])
        res_mat2 = []
        for epsi in mat2:
            a, b, c = epsi[0], np.flip(epsi[1]), epsi[2]
            comb_mat = eigen_compress(np.array([a, b, c]))
            res_mat2.append(comb_mat[0])

        res_mat.append(res_mat2)

    return np.array(res_mat)


def get_epoch_surface(dir_path_, mod_type_, f_name_, model_name_, filter_types_, dataset_name_):
    epochs = np.arange(10, 101, 10)
    data_arr_og = np.empty((3, 10, 25, 25))
    data_arr_adv = np.empty((3, 10, 25, 25))
    density_arr = np.empty((3, 10, 25, 25))
    vals_arr = np.empty((3, 10, 25, 25))

    for fil, fil_type in enumerate(filter_types_):
        for i, e in enumerate(epochs):
            file_name = dir_path_+'/'+mod_type_+'/'+f_name_+'/'+model_name_+'_comb2_'+fil_type+'_'+dataset_name_+'_'+str(e)+'epochs'
            data = load_pickle(file_name)[0]
            for j, epsi in enumerate(data):
                # print(data.keys())
                temp_alpha = 0
                for k, alpha in enumerate(data[epsi]):
                    data_arr_og[fil][i][j][k] = data[epsi][alpha][0]
                    data_arr_adv[fil][i][j][k] = data[epsi][alpha][1]
                    temp_alpha = alpha

                density_arr[fil][i][j] = data[epsi][temp_alpha][2]

    return [data_arr_og, data_arr_adv, density_arr]


def combined_data(data_, filts=3):
    img_plot = data_
    temp_sig_epsi = []
    for i in range(25):
        temp_sig_epoch = []
        for j in range(10):
            temp_sig_filt = []
            for k in range(filts):
                temp_sig_filt.append(img_plot[k][j][i])

            temp_sig_filt = np.array(temp_sig_filt)
            ret_mat = eigen_compress(temp_sig_filt)[0]
            temp_sig_epoch.append(ret_mat)

        temp_sig_epoch = np.array(temp_sig_epoch)
        temp_sig_epsi.append(temp_sig_epoch)

    data_fin = np.array(temp_sig_epsi)
    return data_fin

def normalise_x(x):
  return (x - np.min(x)) / (np.max(x) - np.min(x))

def integrate(f, x):
  return integrate.simps(f, x, even='avg')

def rob_value(og_acc_, adv_acc_, density_0, alpha_, max_dens_):
    alpha_ = np.array(alpha_)
    density_ = density_0 / max_dens_

    # print(density_.shape, og_acc_.shape, adv_acc_.shape, alpha_.shape)

    # if model_type == 'SqueezeNet':
    #   density_ = density_[0][0]
    # elif model_type == 'ResNet50':
    #  density_ = density_[0]
    # density_ = density_

    # if filt == 'Step_maxmin/' and dataset_type == 'CIFAR10':
    #     og_acc_, adv_acc_ = np.flip(og_acc_), np.flip(adv_acc_)
    #     density_ = np.flip(density_)
    #     alpha_ = np.flip(alpha_)

    og_temp = og_acc_[0]
    adv_temp = adv_acc_[0]

    dens_bar = (1 - density_)
    alpha_ = normalise_x(alpha_)

    # a_ = (normalise_x(density_))*test_acc_og_
    # b_ = (normalise_x(density_))*test_acc_adv_
    a_ = -(dens_bar - og_acc_)
    b_ = -(dens_bar - adv_acc_)

    rob_value_a = integrate(a_, alpha_)
    rob_value_b = integrate(b_, alpha_)  # /integrate(b_full, alpha_)

    rob_val = integrate(a_ - b_, alpha_)

    return rob_value_a, rob_value_b, rob_val


dir_path, mod_type, dataset_name, model_name = 'Plot_data/', 'cube_data', 'MNIST',  'ResNet18'
filter_types = ['step_minmax', 'step_maxmin', 'step_pulse_minmax']

f_name = mod_type+'_'+model_name+'_'+dataset_name+'_V2'
# print(os.listdir(dir_path+mod_type+'/'+f_name))

og_data, adv_data, density_data = get_epoch_surface(dir_path, mod_type, f_name, model_name, filter_types, dataset_name)
og_comb = combined_data(og_data)
adv_comb = combined_data(adv_data)
dens_comb = combined_data(density_data, filts=2)
data = adv_comb

alpha = np.linspace(0, 1, 25, endpoint=True)



# print(adv_comb[0][0])
# print("\n")

# mask = adv_comb[adv_comb > 0.35]
# adv_comb[mask] = np.nan
# data = adv_comb

# minmax_data = load_cube_data(dir_path, mod_name, dataset_name, analysis_type, filter_types[0])  # Rerun minmax
# maxmin_data = load_cube_data(dir_path, mod_name, dataset_name, analysis_type, filter_types[1])
# pulse_data = load_cube_data(dir_path, mod_name, dataset_name, analysis_type, filter_types[2])  # pulse data == step_pulse_minmax_
#
# fin_mat = combine_response(minmax_data, maxmin_data, pulse_data)
#
# print(fin_mat.shape)
volume = data
x = np.arange(volume.shape[0])[:, None, None]
y = np.arange(volume.shape[1])[None, :, None]
z = np.arange(volume.shape[2])[None, None, :]
x, y, z = np.broadcast_arrays(x, y, z)

# cube_shape = volume.shape
# space = np.array([*product(range(cube_shape[0]), range(cube_shape[1]), range(cube_shape[2]))])
#
# fig = plt.figure(figsize=(10,10))
# ax = fig.gca(projection='3d')
# p = ax.scatter(space[:,0], space[:,1], space[:,2], marker='s', s=300, alpha=0.5, c=volume.ravel(), vmin=np.min(volume), vmax=np.max(volume), cmap='Spectral')
# plt.colorbar(p, shrink=0.5, aspect=2)
#
# print(len(volume))
# ax.set_xticks(np.arange(0, data.shape[0], 1))
# ax.set_yticks(np.arange(data.shape[1], 0, -1))
# ax.set_zticks([0, 25])
#
# epsi_x_vals = [round(x,2) for x in np.linspace(0, 1, num=25)]
# epoch_y_vals = np.arange(10, 110, 10)
# alpha_z_vals = ['$\\alpha_{0}$', '$\\alpha_{R}$']
#
# ax.set_xticklabels(epsi_x_vals)
# ax.set_yticklabels(epoch_y_vals)
# ax.set_zticklabels(alpha_z_vals)
#
# print(volume.ravel().shape)
#
# ax.set_zlabel('alpha $\\alpha$')
# ax.set_ylabel("Epochs")
# ax.set_xlabel('epsilon ($\epsilon$)')
#
# plt.tight_layout()
# plt.show()
