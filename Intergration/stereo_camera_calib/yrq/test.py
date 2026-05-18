import json
import fcntl
import os
# t_cover = [1,2,3]
with open('./pose_start.json', 'w', encoding='utf-8') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    json.dump(1, f)
    f.flush()
    os.fsync(f.fileno())
    fcntl.flock(f, fcntl.LOCK_UN)
    

# with open('./cover_pose.json', 'r', encoding='utf-8') as f:
#     a = json.load(f)

# print(a)

# import numpy as np

# m = [[1,2,3], [4,5,0], [7,8,9]]
# print(np.mean(m, axis=0))