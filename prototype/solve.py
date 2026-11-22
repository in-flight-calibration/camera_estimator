import argparse, time
from common import *
from scipy.optimize import least_squares

def select(F, per_frame, rng):
    out = []
    for f in F:
        M = f['M']
        if len(M) == 0:
            continue
        M = M[M[:, 4] > 0]
        if len(M) < 20:
            continue
        b = (np.clip(((M[:, 0] + 1) / 2 * 4).astype(int), 0, 3) * 4 + np.clip(((M[:, 1] + 1) / 2 * 4).astype(int), 0, 3))
        order = rng.permutation(len(M)); M = M[order]; b = b[order]
        rank = np.zeros(len(M), int); cnt = {}
        for k, bb in enumerate(b):
            rank[k] = cnt.get(bb, 0); cnt[bb] = rank[k] + 1
        M = M[np.argsort(rank, kind='stable')][:per_frame]
        out.append((f, M))
    return out

def build(sel):
    rows = []
    for f, M in sel:
        R1 = quat_to_R(f['pap']['q']); R2 = quat_to_R(f['ap']['q'])
        c1 = f['pap']['p']; c2 = f['ap']['p']
        for m in M:
            rows.append((m[0], m[1], m[2], m[3], *R1.ravel(), *R2.ravel(), *c1, *c2))
    A = np.array(rows)
    return dict(x1=A[:, 0:2], x2=A[:, 2:4], R1=A[:, 4:13].reshape(-1, 3, 3), R2=A[:, 13:22].reshape(-1, 3, 3), c1=A[:, 22:25], c2=A[:, 25:28])

def unpack(p):
    return rotvec_to_R(p[:3]), p[3], p[4], p[5], p[6], p[7]

def rays(p, D):
    Rm, fx, fy, cx, cy, lam = unpack(p)
    u = (D['x1'][:, 0] - cx) / fx; v = (D['x1'][:, 1] - cy) / fy; s = 1 + lam * (u * u + v * v)
    dc = np.stack([u / s, v / s, np.ones(len(D['x1']))], 1)
    dw = np.einsum('nij,jk,nk->ni', D['R1'], Rm @ N_OPT.T, dc)
    return dc, dw

def residual(p, D, zc):
    Rm, fx, fy, cx, cy, lam = unpack(p)
    dc, dw = rays(p, D)
    if zc is None:
        X = dw
        Xc = np.einsum('ij,nkj,nk->ni', N_OPT @ Rm.T, D['R2'], X)
    else:
        X = D['c1'] + zc[:, None] * dw - D['c2']
        Xc = np.einsum('ij,nkj,nk->ni', N_OPT @ Rm.T, D['R2'], X)
    z = np.where(Xc[:, 2] > 1e-6, Xc[:, 2], 1e-6)
    xu = Xc[:, 0] / z; yu = Xc[:, 1] / z; s = 2 / (1 + np.sqrt(np.maximum(1 - 4 * lam * (xu * xu + yu * yu), 1e-12)))
    u = fx * s * xu + cx - D['x2'][:, 0]; v = fy * s * yu + cy - D['x2'][:, 1]
    return np.concatenate([u, v])

def depths(p, D, T, max_range):
    dc, dw = rays(p, D)
    s = T.raycast(D['c1'], dw, max_range=max_range)
    return s / np.linalg.norm(dw, axis=1)

def report(tag, p, W, H, extra=''):
    Rm, fx, fy, cx, cy, lam = unpack(p)
    hfov = 2 * np.degrees(np.arctan(1 / fx)); vfov = 2 * np.degrees(np.arctan(1 / fy))
    e = R_to_euler(Rm)
    print(f'{tag}: HFOV {hfov:.2f} VFOV {vfov:.2f} fx {fx:.4f} fy {fy:.4f} (fy/fx {fy/fx:.4f}, W/H {W/H:.4f}) cx {cx:+.4f} ({cx*W/2:+.1f}px) cy {cy:+.4f} ({cy*H/2:+.1f}px) mount rpy {e[0]:+.2f} {e[1]:+.2f} {e[2]:+.2f} deg lambda {lam:+.4f} q {np.round(R_to_quat(Rm), 4)} {extra}', flush=True)

def solve(D, T, p0, W, H, huber, max_range, outer, verbose=True, rotation_only=False):
    p = p0.copy()
    for it in range(outer):
        t0 = time.time()
        if rotation_only:
            zc = None; keep = np.ones(len(D['x1']), bool)
        else:
            zc = depths(p, D, T, max_range); keep = np.isfinite(zc)
        Dk = {k: v[keep] for k, v in D.items()}
        r = least_squares(residual, p, args=(Dk, None if zc is None else zc[keep]), loss='huber', f_scale=huber, x_scale='jac', method='trf')
        dp = np.abs(r.x - p).max(); p = r.x
        res = r.fun.reshape(2, -1); rn = np.hypot(res[0] * W / 2, res[1] * H / 2)
        if verbose:
            report(f'  it{it} hits {keep.mean()*100:.0f}% n={keep.sum()}', p, W, H, f'| med res {np.median(rn):.2f}px | {time.time()-t0:.1f}s')
        if dp < 1e-5:
            break
    J = r.jac; s2 = 2 * r.cost / max(1, len(r.fun) - 8)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((8, 8), np.nan)
    return p, cov, keep

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=3000); ap.add_argument('--per-frame', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--hfov', type=float, default=60.0)
    ap.add_argument('--huber', type=float, default=0.005); ap.add_argument('--max-range', type=float, default=3000.0)
    ap.add_argument('--outer', type=int, default=15); ap.add_argument('--rotation-only', action='store_true')
    ap.add_argument('--part', type=str, default='all'); ap.add_argument('--rate', type=str, default=''); ap.add_argument('--data', default='frames11.pkl'); ap.add_argument('--yaw0', type=float, default=0.0); ap.add_argument('--pitch0', type=float, default=0.0)
    a = ap.parse_args()
    d = load(a.data); W, H = d['frames'][0]['size']; W = float(W); H = float(H)
    F = [f for f in d['frames'] if f['complete'] and np.all(np.isfinite(f['ap']['q'])) and np.all(np.isfinite(f['pap']['p'])) and np.all(np.isfinite(f['ap']['p']))]
    if a.part != 'all':
        k, n = map(int, a.part.split('/')); F = F[len(F) * k // n:len(F) * (k + 1) // n]
    if a.rate:
        lo, hi = map(float, a.rate.split(','))
        def rate(f):
            Rr = quat_to_R(f['pap']['q']).T @ quat_to_R(f['ap']['q'])
            return np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
        F = [f for f in F if lo <= rate(f) < hi]
    rng = np.random.default_rng(a.seed)
    if a.frames < len(F):
        F = [F[i] for i in sorted(rng.choice(len(F), a.frames, replace=False))]
    D = build(select(F, a.per_frame, rng)); T = Terrain(load('frames11.pkl')['terrains'])
    fx0 = 1 / np.tan(np.radians(a.hfov / 2)); r0 = np.radians([0, a.pitch0, a.yaw0]); p0 = np.array([*r0, fx0, fx0 * W / H, 0, 0, 0])
    print(f'frames {len(F)} matches {len(D["x1"])}')
    p, cov, keep = solve(D, T, p0, W, H, a.huber, a.max_range, a.outer, rotation_only=a.rotation_only)
    sd = np.sqrt(np.diag(cov))
    report('FINAL', p, W, H)
    print(f'  sd: rotvec {np.degrees(sd[:3]).round(3)} deg, fx {sd[3]:.5f} fy {sd[4]:.5f} cx {sd[5]*W/2:.2f}px cy {sd[6]*H/2:.2f}px lambda {sd[7]:.5f}')
    np.save(f'p_{a.seed}_{a.part.replace("/", "of")}.npy', p)
