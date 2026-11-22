import sys, pickle, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
ns = lambda t: t.sec * 1_000_000_000 + t.nanosec
r = make_reader(open(sys.argv[1], 'rb'), decoder_factories=[DecoderFactory()])
frames = []
for sc, ch, msg, m in r.iter_decoded_messages(topics=['/stream_telemetry/stream_telemetry']):
    def ap(a):
        return dict(ts=ns(a.image_timestamp), q=np.array(a.attitude), w=np.array(a.angular_velocity), p=np.array(a.position), v=np.array(a.velocity), lla=(a.latitude, a.longitude, a.altitude))
    f = m.features
    M = np.array([[k.x1, k.y1, k.x2, k.y2, k.inlier, k.quality] for k in f.matches], dtype=np.float64).reshape(-1, 6)
    frames.append(dict(t=msg.log_time, complete=m.complete, reset=m.stream_info.reset, size=tuple(m.stream_info.size), ts=ns(m.stream_info.image_timestamp), pts=ns(m.prev_stream_info.image_timestamp), ap=ap(m.autopilot_telemetry), pap=ap(m.prev_autopilot_telemetry), hz=(m.horizon.theta, m.horizon.rho, m.horizon.confidence), M=M))
pickle.dump(dict(frames=frames, terrains=[]), open(sys.argv[2], 'wb'))
F = [f for f in frames if f['complete']]
print(len(frames), len(F), set(f['size'] for f in frames))
R = 6371000.0
P = np.array([f['ap']['p'] for f in F]); L = np.array([f['ap']['lla'] for f in F])
lat0 = L[:, 0] - np.degrees(P[:, 0] / R); lon0 = L[:, 1] - np.degrees(P[:, 1] / (R * np.cos(np.radians(L[:, 0]))))
print('ref lat', np.median(lat0), np.std(lat0) * 111e3, 'm; ref lon', np.median(lon0), np.std(lon0) * 68e3, 'm; ref alt', np.median(L[:, 2] + P[:, 2]))
