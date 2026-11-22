import pickle, numpy as np

N_OPT = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])

def quat_to_R(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))

def rotvec_to_R(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_quat(R):
    w = np.sqrt(max(0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])

def R_to_euler(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])

def load(path='frames11.pkl'):
    return pickle.load(open(path, 'rb'))

class Terrain:
    def __init__(self, terrains):
        step = terrains[0]['x'][1]
        x0 = min(t['x'][0] for t in terrains); x1 = max(t['x'][2] for t in terrains)
        y0 = min(t['y'][0] for t in terrains); y1 = max(t['y'][2] for t in terrains)
        nx = int(round((x1 - x0) / step)) + 1; ny = int(round((y1 - y0) / step)) + 1
        Z = np.full((nx, ny), np.nan)
        for t in terrains:
            n = int(round((t['x'][2] - t['x'][0]) / step)) + 1; m = int(round((t['y'][2] - t['y'][0]) / step)) + 1
            i = int(round((t['x'][0] - x0) / step)); j = int(round((t['y'][0] - y0) / step))
            Z[i:i + n, j:j + m] = t['z'].reshape(n, m)
        self.Z, self.x0, self.y0, self.step = Z, x0, y0, step
        self.zmin, self.zmax = np.nanmin(Z), np.nanmax(Z)

    def z(self, x, y):
        fi = (x - self.x0) / self.step; fj = (y - self.y0) / self.step
        i = np.floor(fi).astype(int); j = np.floor(fj).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < self.Z.shape[0] - 1) & (j < self.Z.shape[1] - 1)
        i = np.where(ok, i, 0); j = np.where(ok, j, 0)
        a = fi - i; b = fj - j
        Z = self.Z
        z = (Z[i, j] * (1 - a) * (1 - b) + Z[i + 1, j] * a * (1 - b) + Z[i, j + 1] * (1 - a) * b + Z[i + 1, j + 1] * a * b)
        return np.where(ok, z, np.nan)

    def raycast(self, c, d, max_range=3000.0, ds=4.0):
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        n = len(c)
        s_hit = np.full(n, np.nan)
        down = d[:, 2] > 1e-3
        s0 = np.where(down, np.maximum(0, (self.zmin - c[:, 2]) / np.where(down, d[:, 2], 1)), np.inf)
        s1 = np.where(down, np.minimum(max_range, (self.zmax - c[:, 2]) / np.where(down, d[:, 2], 1)), -np.inf)
        idx = np.nonzero(down & (s0 < s1))[0]
        if len(idx) == 0:
            return s_hit
        c, d, s0, s1 = c[idx], d[idx], s0[idx], s1[idx]
        steps = int(np.ceil((s1 - s0).max() / ds)) + 1
        prev_s = s0.copy()
        p = c + prev_s[:, None] * d
        prev_f = p[:, 2] - self.z(p[:, 0], p[:, 1])
        found = np.zeros(len(idx), bool); lo = np.zeros(len(idx)); hi = np.zeros(len(idx))
        dead = ~(prev_f < 0)
        for k in range(1, steps + 1):
            act = ~found & ~dead
            if not act.any():
                break
            s = np.minimum(s0 + k * ds, s1)
            p = c + s[:, None] * d
            f = p[:, 2] - self.z(p[:, 0], p[:, 1])
            hit = act & (f >= 0)
            lo[hit] = prev_s[hit]; hi[hit] = s[hit]; found |= hit
            dead |= act & (np.isnan(f) | (s >= s1) & ~hit)
            prev_s = s
        for _ in range(12):
            m = (lo + hi) / 2
            p = c + m[:, None] * d
            f = p[:, 2] - self.z(p[:, 0], p[:, 1])
            below = f >= 0
            hi = np.where(below, m, hi); lo = np.where(below, lo, m)
        s_hit[idx[found]] = hi[found]
        return s_hit
