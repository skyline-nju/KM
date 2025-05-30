import sys
import numpy as np
from gsd import hoomd, fl
import os


def duplicate(s, nx: int, ny: int):
    N = s.particles.N * nx * ny
    lx = s.configuration.box[0]
    ly = s.configuration.box[1]
    Lx, Ly = lx * nx, ly * ny
    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            beg = (j * nx + i) * s.particles.N
            end = beg + s.particles.N
            pos[beg:end, 0] = s.particles.position[:, 0] + lx / 2 + i * lx
            pos[beg:end, 1] = s.particles.position[:, 1] + ly / 2 + j * ly
            pos[beg:end, 2] = s.particles.position[:, 2]
            charge[beg:end] = s.particles.charge
            mass[beg:end] = s.particles.mass
    pos[:, 0] -= Lx / 2
    pos[:, 1] -= Ly / 2
    s2 = hoomd.Frame()
    s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    s2.particles.N = N
    s2.particles.position = pos
    s2.particles.charge = charge
    s2.particles.mass = mass
    s2.configuration.step = 0
    return s2


def create_spin_waves(angle, L, rho0=1, n_period=1):
    # angle <= pi/4
    s = hoomd.Frame()
    s.configuration.box = [L, L, 1, 0, 0, 0]
    N = int(L ** 2 * rho0)
    s.particles.N = N
    s.configuration.step = 0

    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)

    pos[:, 0] = np.random.rand(N) * L
    pos[:, 1] = np.random.rand(N) * L
    pos[:, 2] = (np.random.rand(N) - 0.5)  * np.pi * 2

    z = pos[:, 0] + pos[:, 1] * np.tan(angle)
    z[z>= L] -= L
    charge = n_period * z / L
    charge = charge - np.floor(charge)

    charge *= np.pi * 2
    charge[charge >= np.pi] -= 2 * np.pi

    pos[:, 0] -= L / 2
    pos[:, 1] -= L / 2
    s.particles.position = pos
    s.particles.charge = charge
    s.particles.mass = mass
    return s


def get_snap(fname, i_frame=-1):
    def read_fl(fname, i_frame):
        with fl.open(name=fname, mode="r") as f:
            if i_frame < 0:
                i_frame += f.nframes
            s.configuration.box = f.read_chunk(frame=0, name="configuration/box")
            if f.chunk_exists(frame=i_frame, name="configuration/step"):
                s.configuration.step = f.read_chunk(frame=i_frame, name="configuration/step")[0]
                print(s.configuration.step)
            else:
                if i_frame == 0:
                    s.configuration.step = 0
                else:
                    print("Error, cannot find step for frame =", i_frame)
                    sys.exit()
            s.particles.N = f.read_chunk(frame=i_frame, name="particles/N")[0]
            s.particles.position = f.read_chunk(frame=i_frame, name="particles/position")
            s.particles.charge = f.read_chunk(frame=i_frame, name="particles/charge")
            s.particles.mass = f.read_chunk(frame=i_frame, name="particles/mass")    
            return s

    s = hoomd.Frame()
    with hoomd.open(name=fname_in, mode='r') as fin:
        try:
            nframes = len(fin)
            if i_frame < 0:
                i_frame == nframes
            s = fin[i_frame]
        except IndexError:
            print("Failed to open", fname, "in the hoomd mode")
            s = read_fl(fname, i_frame)
    return s


def create_spin_waves_along_y(Lx, Ly, rho0=1, n_period=1):
    s = hoomd.Frame()
    s.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    N = int(Lx * Ly * rho0)
    s.particles.N = N
    s.configuration.step = 0

    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)

    pos[:, 0] = np.random.rand(N) * Lx
    pos[:, 1] = np.random.rand(N) * Ly
    pos[:, 2] = (np.random.rand(N) - 0.5)  * np.pi * 2

    charge = n_period *  pos[:, 1] / Ly
    charge = charge - np.floor(charge)

    charge *= np.pi * 2
    charge[charge >= np.pi] -= 2 * np.pi

    pos[:, 0] -= Lx / 2
    pos[:, 1] -= Ly / 2
    s.particles.position = pos
    s.particles.charge = charge
    s.particles.mass = mass
    return s


def scale(s, nx: int, ny: int, eps=0):
    lx = s.configuration.box[0]
    ly = s.configuration.box[1]
    Lx, Ly = lx * nx, ly * ny
    if isinstance(nx, int) and isinstance(ny, int):
        N = s.particles.N * nx * ny
        pos = np.zeros((N, 3), dtype=np.float32)
        type_id = np.zeros(N, dtype=np.uint32)
        for i in range(nx * ny):
            beg = i * s.particles.N
            end = beg + s.particles.N
            pos[beg:end, 0] = s.particles.position[:, 0] * nx
            pos[beg:end, 1] = s.particles.position[:, 1] * ny
            pos[beg:end, 2] = s.particles.position[:, 2]
            type_id[beg:end] = s.particles.typeid
        if nx > 1:
            pos[:, 0] += (np.random.rand(N) - 0.5) * eps * nx
            mask = pos[:, 0] < Lx/2
            pos[:, 0][mask] += Lx
            mask = pos[:, 0] >= Lx/2
            pos[:, 0][mask] -= Lx
        if ny > 1:
            pos[:, 1] += (np.random.rand(N) - 0.5) * eps * ny
            mask = pos[:, 1] < Ly/2
            pos[:, 1][mask] += Ly
            mask = pos[:, 1] >= Ly/2
            pos[:, 1][mask] -= Ly
        s2 = hoomd.Frame()
        s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
        s2.particles.N = N
        s2.particles.position = pos
        s2.particles.typeid = type_id
        s2.particles.types = s.particles.types
        s2.configuration.step = 0
    else:
        s.particles.position[:, 0] *= nx
        s.particles.position[:, 1] *= ny
        s.configuration.box = [Lx, Ly, 1, 0, 0, 0]

        mask_A = s.particles.typeid == 0
        mask_B = s.particles.typeid == 1
        rho_A = np.sum(mask_A) / (lx * ly)
        rho_B = np.sum(mask_B) / (lx * ly)
        print("rho_A=", rho_A, "rho_B=", rho_B)
        s2 = adjust_density(s, rho_A, rho_B, mode="copy")
    return s2


if __name__ == "__main__":
    folder = r"/mnt/sda/active_KM/snap"
    # folder = "build/data"
    basename = "L2048_2048_r1_v1_T0.1_s0.1_D0.1000_h0.1_S3000.gsd"

    fname_in = f"{folder}/{basename}"

    snap = get_snap(fname_in, 81)

    fname_out = f"{folder}/L2048_2048_r1_v1_T0.1_s0.1_D0.1000_h0.1_S2082.gsd"
    with hoomd.open(name=fname_out, mode='w') as fout:
        # snap_new = duplicate(snap, 2, 2)
        # snap_new = create_spin_waves_along_y(2048, 128, 2.5)
        # snap_new = create_spin_waves(0, 128, 2.5, 2)
        snap.configuration.step = 0
        fout.append(snap)

