from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import interpolate, signal, ndimage
from scipy.integrate import simps, dblquad


def get_data(file):
    accuracy_vals = np.loadtxt(file, dtype=float)

    x = np.arange(len(accuracy_vals))
    y = np.arange(len(accuracy_vals[0]))

    y_mesh, x_mesh = np.meshgrid(y, x)

    return x_mesh, y_mesh, accuracy_vals


def get_raw_data(file):
    z = np.loadtxt(file, dtype=float)

    x = np.arange(len(z))
    y = np.arange(len(z[0]))

    return x, y, z


# print(np.shape(X), np.shape(Y), np.shape(Z))
def plot1(x, y, z, _x, _y, _z):
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cmap='Spectral', rstride=1, cstride=1, alpha=0.8, antialiased=True)

    plt.title("Un-filtered Surface Plot")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_zlabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()


def plot2(x, y, z, _x, _y, _z):
    xnew = x
    ynew = y
    tck = interpolate.bisplrep(x, y, _z, s=0.5)
    znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.plot_surface(xnew, ynew, znew, cmap='Spectral', rstride=1, cstride=1, alpha=None, antialiased=True)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_zlabel("Accuracy (%)")

    plt.show()


def plot3(x, y, z, x_ext, y_ext, z_filtered):
    # Subplot 1
    xnew, ynew = x, y
    tck = interpolate.bisplrep(x, y, z_filtered, s=0.1)
    znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(x, y, z, cmap='Spectral', rstride=1, cstride=1, alpha=0.8, antialiased=False)
    ax2.plot_surface(xnew, ynew, znew, cmap='Spectral', rstride=1, cstride=1, alpha=0.8, antialiased=True)

    def on_move(event):
        if event.inaxes == ax:
            if ax.button_pressed in ax._rotate_btn:
                ax2.view_init(elev=ax.elev, azim=ax.azim)
            elif ax.button_pressed in ax._zoom_btn:
                ax2.set_xlim3d(ax.get_xlim3d())
                ax2.set_ylim3d(ax.get_ylim3d())
                ax2.set_zlim3d(ax.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax.set_xlim3d(ax2.get_xlim3d())
                ax.set_ylim3d(ax2.get_ylim3d())
                ax.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_zlabel("Accuracy (%)")

    # set limits fro ax
    ax.set_xlim3d(0, len(x) + 1)
    # ax.set_ylim3d(min(map(min, y))-0.1, max(map(max, y))+0.1)
    # ax.set_zlim3d(min(map(min, z))-0.1, max(map(max, y))+0.1)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Probability")
    ax2.set_zlabel("Accuracy (%)")

    # set limits for ax1
    ax2.set_xlim3d(0, len(xnew))
    # ax2.set_ylim3d(min(map(min, ynew))-0.1, max(map(max, znew))+0.1)
    # ax2.set_zlim3d(min(map(min, znew))-0.1, max(map(max, znew))+0.1)

    ax.set_title("Robustness Surface")
    ax2.set_title("Interpolated Surface With s=0.9")

    # plt.tight_layout()
    plt.show()


def plot4(x, y, z, _x, _y, _z):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x, y, z, cmap='Spectral', rstride=1, cstride=1, alpha=0.9, antialiased=True)
    ax2.plot_surface(_x, _y, _z, cmap='bone', rstride=1, cstride=1, alpha=0.9, antialiased=True)

    def on_move(event):
        if event.inaxes == ax:
            if ax.button_pressed in ax._rotate_btn:
                ax2.view_init(elev=ax.elev, azim=ax.azim)
            elif ax.button_pressed in ax._zoom_btn:
                ax2.set_xlim3d(ax.get_xlim3d())
                ax2.set_ylim3d(ax.get_ylim3d())
                ax2.set_zlim3d(ax.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax.set_xlim3d(ax2.get_xlim3d())
                ax.set_ylim3d(ax2.get_ylim3d())
                ax.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_zlabel("Accuracy (%)")

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Probability")
    ax2.set_zlabel("Accuracy (%)")

    plt.show()


def filter_sigs(sig, c_off, max_fs):
    sig = sig.T
    low = c_off / (max_fs * 0.5)
    [b, a] = signal.butter(2, low, 'low')
    empty_array = np.zeros_like(sig)

    for i in range(len(sig)):
        z_ = signal.filtfilt(b, a, sig[i, :])
        empty_array[i] = z_

    empty_array = empty_array.T
    return empty_array


def simps_rule(func, a, b, n=50):
    if n % 2 == 1:
        raise ValueError("N must be an even integer")
    dx = (b - 1) / n
    xx = np.linspace(a, b, n + 1)
    yy = func(xx)
    ss = (dx / 3) * np.sum(yy[0:-1:2] + 4 * yy[1::2] + yy[2::2])
    return ss


def double_integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    a_internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(a_internal)
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))


def integral_difference(x_array, y_array, z_array, rr):
    robust_surface = np.ones_like(z_array) * rr
    full_surface = np.ones_like(z_array) * 100
    rms_sur = np.ones_like(z_array)*(np.sqrt(np.mean(z_array**2)))

    # robust_surface = z_array - robust_surface
    x_min, x_max, n_points_x = (0, len(x_array), len(x_array))
    y_min, y_max, n_points_y = (0, len(y_array), len(y_array))

    accuracy_surface_int = double_integral(x_min, x_max, y_min, y_max, n_points_x, n_points_y, rms_sur)
    robust_surface_int = double_integral(x_min, x_max, y_min, y_max, n_points_x, n_points_y, robust_surface)
    full_surface_int = double_integral(x_min, x_max, y_min, y_max, n_points_x, n_points_y, full_surface)

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(x_array, y_array, robust_surface, cmap='Spectral', rstride=1,
    #                 cstride=1, alpha=0.75, antialiased=True)
    # ax.plot_surface(x_array, y_array, rms_sur, color='orange', rstride=1, cstride=1,
    #                 alpha=None, antialiased=True)
    # ax.plot_surface(x_array, y_array, full_surface, cmap='coolwarm', rstride=1, cstride=1,
    #                 alpha=None, antialiased=True)

    # ax.set_xlabel("Layer")
    # ax.set_ylabel("Probability")
    # ax.set_zlabel("Accuracy (%)")
    # plt.show()

    return accuracy_surface_int, robust_surface_int, full_surface_int, z_array


def integral_simps(x_array, y_array, z_array, rr):
    robust_surface = np.ones_like(z_array) * rr
    full_surface = np.ones_like(z_array)
    rms_sur = np.ones_like(z_array)*(np.sqrt(np.mean(z_array**2)))

    # print("z_array rob score", (np.sqrt(np.mean(z_array**2))-np.min(z_array))/(np.max(z_array)-np.min(z_array)))

    robust_surface_int = simps([simps(zz_x, y_array[0, :]) for zz_x in robust_surface], x_array[:, 0])
    full_surface_int = simps([simps(zz_x, y_array[0, :]) for zz_x in full_surface], x_array[:, 0])
    rms_surface_int = simps([simps(zz_x, y_array[0, :]) for zz_x in rms_sur], x_array[:, 0])

    return rms_surface_int, robust_surface_int, full_surface_int


def ones_like_array(array, magnitude):
    return np.ones_like(array)*magnitude


def plot_rms(x, y, z):
    rms_sur = np.ones_like(z)*(np.sqrt(np.mean(z**2)))
    des_acc = np.arange(np.max(z), 0.1, -1)
    int_diff = []

    for i in range(len(des_acc)):
        a, b, c, _ = integral_difference(x, y, rms_sur, des_acc[i])
        print(a, b)
        int_diff.append(abs(b-a)/c)

    plt.plot(int_diff)
    plt.ylabel('Surface Integral Difference (Normalised Magnitude)')
    plt.xlabel('Desired Accuracy')
    plt.show()

    # fig = plt.figure(figsize=(10, 15))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot_surface(x, y, z, cmap='Spectral', rstride=1, cstride=1, alpha=0.8, antialiased=True)
    # ax.plot_surface(x, y, rms_sur, cmap='Spectral', rstride=1, cstride=1, alpha=0.95, antialiased=True)
    #
    # plt.title("Un-filtered Surface Plot")
    #
    # ax.set_xlabel("Layer")
    # ax.set_ylabel("Probability")
    # ax.set_zlabel("Accuracy (%)")
    #
    # plt.tight_layout()
    # plt.show()


# layer_1_av, layer_2_av = "Layer1_robustness\\test_4.txt", "Layer2_robustness\\test_7.txt"
# layer_1, layer_2 = "Layer1_robustness\\test_3.txt", "Layer2_robustness\\test_0.txt"
#
# X, Y, Z = get_data(file=layer_2_av)
# Y = (Y * 0.05) + 0.1
# tck = interpolate.bisplrep(X, Y, Z, s=1)
# Z = interpolate.bisplev(X[:, 0], Y[0, :], tck)

# XX, YY, ZZ = get_data(file=layer_2_av)
# YY = (YY * 0.05) + 0.1
# tck = interpolate.bisplrep(XX, YY, ZZ, s=1)
# ZZ = interpolate.bisplev(XX[:, 0], YY[0, :], tck)

# fs = 36
# cutoff = 7
# Z_ = filter_sigs(ZZ, cutoff, fs)
# tck = interpolate.bisplrep(XX, YY, ZZ, s=0.9)
# znew = interpolate.bisplev(X[:, 0], Y[0, :], tck)

# integral_difference(X, Y, Z, 98.5)
# robust_surface = ones_like_array(Z, 98.5)
# rms_sig = ones_like_array(Z, np.sqrt(np.mean(Z*Z)))
#
# ans_a, ans_b = integral_difference(X, Y, Z, 98.5)
# print(ans_a, ans_b)
# ans_a, ans_b = integral_difference(X, Y, rms_sig, 98.5)
# print(ans_a, ans_b)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='coolwarm')
# ax.plot_surface(X, Y, rms_sig, cmap='coolwarm')
# plt.show()

# XX, YY, ZZ = get_data(file=layer_1_av)
# YY = (YY * 0.05) + 0.1
#
# plot4(X, Y, Z, XX, YY, ZZ)
