import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from add_line import add_line


if __name__ == "__main__":
    Lx = Ly = 4096
    dx = 4
    T = 0.1
    sigma = 0.1
    seed = 3000
    n = int(Lx / dx)
    qx = np.fft.fftfreq(n, d=dx/(2 * np.pi))
    qy = np.fft.fftfreq(n, d=dx/(2 * np.pi))

    q_radius = qx[:n//2]

    qx = np.fft.fftshift(qx)
    qy = np.fft.fftshift(qy)
    qx_ij, qy_ij = np.meshgrid(qx, qy)
    q_module = np.sqrt(qx_ij **2 + qy_ij ** 2)

    x = np.linspace(-n//2 * dx, n//2 * dx, n, endpoint=False)
    y = np.linspace(-n//2 * dx, n//2 * dx, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    r_module = np.sqrt(xx ** 2 + yy ** 2)

    radius = x[n//2:]

    folder = "/mnt/sda/active_KM/snap"
    fname_in = f"{folder}/cg_dx{dx}/L{Lx}_{Ly}_r1_v1_T{T:g}_s{sigma:g}_D0.0000_h0.1_S{seed:d}.npz"
    with np.load(fname_in, "rb") as data:
        ux, uy, num = data["ux"], data["uy"], data["num"]

        mask = num > 0
        ux[mask] /= num[mask]
        uy[mask] /= num[mask]
        nframes = ux.shape[0]
        Sq_u_t = np.zeros((nframes, q_radius.size - 1))
        Cr_u_t = np.zeros((nframes, radius.size - 1))

        for i_frame in range(nframes):

            ux_q = np.fft.fft2(ux[i_frame], norm="ortho")
            uy_q = np.fft.fft2(uy[i_frame], norm="ortho")
        
            Sq_u = ux_q * ux_q.conj() + uy_q * uy_q.conj()

            C_u = np.fft.ifft2(Sq_u)

            Sq_u = np.fft.fftshift(Sq_u).real
            C_u = np.fft.ifftshift(C_u).real

            Sq_u_radius = np.zeros(q_radius.size - 1)
            Cr_u_radius = np.zeros(radius.size - 1)

            for j in range(Sq_u_radius.size):
                mask = np.logical_and(q_module >= q_radius[j], q_module < q_radius[j+1])
                Sq_u_radius[j] = np.sum(Sq_u[mask])


            for j in range(radius.size - 1):
                mask = np.logical_and(r_module >= radius[j], r_module < radius[j+1])
                Cr_u_radius[j] = np.sum(C_u[mask]) / np.sum(mask)

            Sq_u_t[i_frame] = Sq_u_radius
            Cr_u_t[i_frame] = Cr_u_radius
            rr = (radius[1:] + radius[:-1]) / 2
            # plt.plot(rr, Cr_u_radius)
            # print(rr)
            # plt.imshow(C_u, origin="lower", norm=colors.LogNorm())
            # plt.show()
            # plt.close(0)

        beg_frame = 80
        qq = (q_radius[1:] + q_radius[:-1]) / 2
        Sq_u_m = np.mean(Sq_u_t[beg_frame:], axis=0)

        rr = (radius[1:] + radius[:-1]) / 2
        Cr_u_m = np.mean(Cr_u_t[beg_frame:], axis=0)


        basename = os.path.basename(fname_in)
        fname_out = f"{folder}/corr_func/{basename}"

        np.savez_compressed(fname_out, q=qq, Sq=Sq_u_m, r=rr, Cr=Cr_u_m)
