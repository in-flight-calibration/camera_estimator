import sys, pickle, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag = sys.argv[1]; out = sys.argv[2]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
ns = lambda t: t.sec * 1_000_000_000 + t.nanosec
frames, terrains = [], []
while r.has_next():
    topic, data, t = r.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    if topic.endswith('terrain'):
        terrains.append(dict(t=t, ref=(m.ref_lat, m.ref_lon, m.ref_alt), x=np.array(m.x), y=np.array(m.y), z=np.array(m.z, dtype=np.float32)))
        continue
    def ap(a):
        return dict(ts=ns(a.image_timestamp), q=np.array(a.attitude), w=np.array(a.angular_velocity), p=np.array(a.position), v=np.array(a.velocity), ref=(a.ref_lat, a.ref_lon, a.ref_alt))
    f = m.features
    M = np.array([[k.x1, k.y1, k.x2, k.y2, k.inlier, k.quality] for k in f.matches], dtype=np.float64).reshape(-1, 6)
    frames.append(dict(t=t, complete=m.complete, reset=m.stream_info.reset, size=tuple(m.stream_info.size),
        ts=ns(m.stream_info.image_timestamp), pts=ns(m.prev_stream_info.image_timestamp),
        fts=(ns(f.image_timestamp[0]), ns(f.image_timestamp[1])),
        ap=ap(m.autopilot_telemetry), pap=ap(m.prev_autopilot_telemetry),
        hz=(m.horizon.theta, m.horizon.rho, m.horizon.confidence), M=M))
pickle.dump(dict(frames=frames, terrains=terrains), open(out, 'wb'))
print(len(frames), len(terrains))
