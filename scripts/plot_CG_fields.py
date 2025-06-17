import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def untangle_1D(theta):
    phase = 0
    theta_new = np.zeros_like(theta)
    theta_new[0] = theta[0]
    for i in range(1, theta.size):
        dtheta = theta[i] - theta[i-1]
        if dtheta < -np.pi:
            phase += 2 * np.pi
        elif dtheta >= np.pi:
            phase -= 2 * np.pi
        theta_new[i] = theta[i] + phase
    return theta_new


def untangle_2D(theta):
    phase_y = 0
    theta_new = np.zeros_like(theta)

    nrows, ncols = theta.shape
    theta_new[0] = untangle_1D(theta[0])
    for j in range(1, nrows):
        dtheta_y = theta[j , 0] - theta[j-1, 0]
        if dtheta_y < -np.pi:
            phase_y += 2 * np.pi
        elif dtheta_y >= np.pi:
            phase_y -= 2 * np.pi
        theta_new[j] = untangle_1D(theta[j]) + phase_y
    return theta_new


def verify_untangled_angles(theta_untangled):
    nrows, ncols = theta_untangled.shape
    x = np.arange(ncols) + 0.5

    for j in range(nrows):
        dtheta = theta_untangled[j, -1] - theta_untangled[j, 0]
        if dtheta < -np.pi or dtheta >= np.pi:
            plt.plot(x, theta_untangled[j])
            plt.show()
            plt.close()


def get_untangled_angles(ux, uy, sigma=0):
    if sigma == 0:
        theta = np.arctan2(uy, ux)
        theta_untangled = untangle_2D(theta)

    return theta_untangled


def show_untangled_theta():
    folder = "/mnt/sda/active_KM/snap/cg_dx4"
    L = 4096
    rho0 = 1
    v0 = 1
    T = 0.1
    sigma = 0.1
    D = 0.1
    h = 0.1
    seed = 3000
    fname = f"{folder}/L{L:d}_{L:d}_r{rho0:g}_v{v0:g}_T{T:g}_s{sigma:g}_D{D:.4f}_h{h:g}_S{seed}.npz"

    with np.load(fname, "r") as data:
        ux, uy, num = data["ux"], data["uy"], data["num"]
    
        nframes, nrows, ncols = ux.shape

        beg_frame = 40

        for i_frame in range(beg_frame, nframes):
            theta = np.arctan2(uy[i_frame], ux[i_frame])
            ux_new = gaussian_filter(ux[i_frame], sigma=3, mode="wrap")
            uy_new = gaussian_filter(uy[i_frame], sigma=3, mode="wrap")
            theta_new = np.arctan2(uy_new, ux_new)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            ax1.imshow(theta, origin="lower", cmap="hsv")
            ax2.imshow(theta_new, origin="lower", cmap="hsv")
            plt.show()
            plt.close()
            theta_untangled = untangle_2D(theta)
            verify_untangled_angles(theta_untangled)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.8), constrained_layout=True)
            extent = [0, L, 0, L]
            im1 = ax1.imshow(theta, origin="lower", cmap="hsv", extent=extent)
            im2 = ax2.imshow(theta_untangled, origin="lower", extent=extent)
            cb1 = fig.colorbar(im1, ax=ax1, orientation="horizontal")
            cb2 = fig.colorbar(im2, ax=ax2, orientation="horizontal")
            cb1.set_label(r"$\theta(\mathbf{r})$", fontsize="xx-large")
            cb2.set_label(r"Unwrapped $\theta(\mathbf{r})$", fontsize="xx-large")
            plt.show()
            plt.close()


def show_coarse_grained_theta():
    folder = "/mnt/sda/active_KM/snap/cg_dx4"
    L = 1024
    rho0 = 1
    v0 = 1
    T = 0.5
    sigma = 0.
    D = 0.
    h = 0.1
    seed = 1000
    fname = f"{folder}/L{L:d}_{L:d}_r{rho0:g}_v{v0:g}_T{T:g}_s{sigma:g}_D{D:.4f}_h{h:g}_S{seed}.npz"

    with np.load(fname, "r") as data:
        ux, uy, num = data["ux"], data["uy"], data["num"]
    
        nframes, nrows, ncols = ux.shape

        beg_frame = 1550

        for i_frame in range(beg_frame, nframes):
            theta = np.arctan2(uy[i_frame], ux[i_frame])
            ux_new = gaussian_filter(ux[i_frame], sigma=3, mode="wrap")
            uy_new = gaussian_filter(uy[i_frame], sigma=3, mode="wrap")
            theta_new = np.arctan2(uy_new, ux_new)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
            ax.imshow(theta_new, origin="lower", cmap="hsv", extent=[0, L, 0, L])
            ax.set_title(r"$T=%g$" % T, fontsize="xx-large")
            plt.show()
            plt.close()

if __name__ == "__main__":
    # show_untangled_theta()
    show_coarse_grained_theta()