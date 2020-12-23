import numpy as np
a = [1, 1, 1, 1, 0, 0]
arr = np.array(a)
idx = [1,3]
if arr[idx].all() == 1:
    print('ok')