import numpy as np
import matplotlib.pyplot as plt
from gsd import fl
import os
import glob
import sys
import gc
from coarse_grain import get_para


def detect_defects(theta):
    def get_diff_angle(i1, j1, i2, j2):
        theta1 = theta[j1 % ny, i1 % nx]
        theta2 = theta[j2 % ny, i2 % nx]
        dtheta = theta2 - theta1
        if dtheta >= np.pi:
            dtheta -= 2 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2 * np.pi
        return dtheta

    ny, nx = theta.shape

    x = []
    y = []
    charge = []

    for j in range(ny):
        for i in range(nx):
            delta_theta = get_diff_angle(i, j, i+1, j) + get_diff_angle(i+1, j, i+1, j+1) + get_diff_angle(i+1, j+1, i, j+1) + get_diff_angle(i, j+1, i, j)
            my_charge = int(np.round(delta_theta / (2 * np.pi)))
            if my_charge != 0:
                charge.append(my_charge)
                x.append(i + 0.5)
                y.append(j + 0.5)
    return np.array(x), np.array(y), np.array(charge)


def detect_defects_fast(theta):
    def get_diff_angle(theta1, theta2):
        dtheta = theta2 - theta1
        mask = dtheta  >= np.pi
        dtheta[mask] -= 2 * np.pi
        mask = dtheta < -np.pi
        dtheta[mask] += 2 * np.pi
        return dtheta

    theta_left = np.roll(theta, (1, 0), axis=(1, 0))
    theta_upper_left = np.roll(theta, (1, 1), axis=(1, 0))
    theta_upper = np.roll(theta, (0, 1), axis=(1, 0))

    delta_theta = get_diff_angle(theta, theta_left) + get_diff_angle(theta_left, theta_upper_left) + \
        get_diff_angle(theta_upper_left, theta_upper) + get_diff_angle(theta_upper, theta)
    charge_field = np.round(delta_theta / (2 * np.pi)).astype(int)

    ny, nx = charge_field.shape
    x_1D = np.arange(nx) + 0.5
    y_1D = np.arange(ny) + 0.5
    xx, yy = np.meshgrid(x_1D, y_1D)

    mask = charge_field != 0
    x = xx[mask]
    y = yy[mask]
    charge = charge_field[mask]
    return x, y, charge


def show_defects(fname, beg_frame=None, savefig=False, fmt=".jpg", dx=4):
    para = get_para(fname, suffix=".npz")
    Lx, Ly = para["Lx"], para["Ly"]

    if Lx == 4096:
        figsize = (9, 8)

    if savefig == True:
        outdir = f"/mnt/sda/active_KM/snap/imgs/{os.path.basename(fname).rstrip(".npz")}"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
    with np.load(fname, "r") as data:
        t = data["t"]
        nframes = t.size
        if beg_frame is None:
            snaps = glob.glob(f"{outdir}/*{fmt}")
            beg_frame = len(snaps)
        elif beg_frame < 0:
            beg_frame += nframes
        
        half_dx = dx * 0.5
        extent = [-half_dx, Lx-half_dx, -half_dx, Ly-half_dx]

        for i_frame in range(beg_frame, nframes):
            print("frame %g/%g" % (i_frame, nframes))
            theta = np.arctan2(data["uy"][i_frame], data["ux"][i_frame])
            x, y, charge = detect_defects_fast(theta)
            x *= dx
            y *= dx

            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
            theta[theta < 0] += np.pi * 2
            im = ax.imshow(theta, cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi, extent=extent)

            for j in range(charge.size):
                if charge[j] == 1:
                    mk = "ko"
                elif charge[j] == -1:
                    mk = "ks"
                ax.plot(x[j], y[j], mk, fillstyle="none", ms=8, mew=2)
            
            if charge.size > 0:
                n_tot = charge.size
                n_plus = np.sum(charge == 1)
                n_mins = np.sum(charge == -1)
            else:
                n_tot = n_plus = n_mins = 0
            ax.set_title(r"Defect number= %d with $n_{+1}=%d, n_{-1}=%g, t=%g$" % (n_tot, n_plus, n_mins, t[i_frame]), fontsize="xx-large")
            fig.colorbar(im)

            if savefig:
                plt.savefig(f"{outdir}/{i_frame:06d}{fmt}")
            else:
                plt.show()
            plt.clf()
            plt.close()
            del theta
            gc.collect()

    
def update_imgs(Lx, Ly=None, dx=4):
    import matplotlib
    matplotlib.use("Agg")

    if Ly is None:
        Ly = Lx
    
    folder = f"/mnt/sda/active_KM/snap/cg_dx{dx:g}"
    pat = f"{folder}/L{Lx:g}_{Ly:g}_*.npz"
    files = glob.glob(pat)

    for f in files:
        show_defects(f, beg_frame=None, savefig=True, fmt=".jpg", dx=dx)

if __name__ == "__main__":
    update_imgs(Lx=4096)