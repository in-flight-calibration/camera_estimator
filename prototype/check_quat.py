from common import *
d = load(); F = [f for f in d['frames'] if f['complete']]
R = quat_to_R(np.array([f['ap']['q'] for f in F])); V = np.array([f['ap']['v'] for f in F])
yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0])); course = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
e = (yaw - course + 180) % 360 - 180; print('yaw-course pct', np.percentile(e, [5, 50, 95]))
eu = np.array([R_to_euler(r) for r in R]); print('roll pct', np.percentile(eu[:, 0], [5, 50, 95]), 'pitch pct', np.percentile(eu[:, 1], [5, 50, 95]))
# relative rotation vs integrated angular velocity
P = quat_to_R(np.array([f['pap']['q'] for f in F])); dt = np.array([(f['ts'] - f['pts']) / 1e9 for f in F])
Rr = np.einsum('nji,njk->nik', P, R)
w = np.array([f['ap']['w'] for f in F])
rv = np.stack([Rr[:, 2, 1] - Rr[:, 1, 2], Rr[:, 0, 2] - Rr[:, 2, 0], Rr[:, 1, 0] - Rr[:, 0, 1]], 1) / 2 / dt[:, None]
print('corr body rate vs dq', [np.corrcoef(rv[:, k], w[:, k])[0, 1] for k in range(3)])
T = Terrain(d['terrains']); print('grid', T.Z.shape, np.isnan(T.Z).mean())
p = np.array([f['ap']['p'] for f in F]); agl = T.z(p[:, 0], p[:, 1]) - p[:, 2]; print('AGL pct', np.percentile(agl, [5, 50, 95]))
