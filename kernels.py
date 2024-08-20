from tinygrad import Tensor

DIMS = [2,2,2]
NDIMS = len(DIMS)
K = 2

kernels = []
for i in range(NDIMS):
    padding = tuple([(0,K-1) if j != i else (0,0) for j in range(NDIMS-1,-1,-1)]) # ((pad to K) ... don't pad ... (pad to K))
    line_shape = [K if j == i else 1 for j in range(NDIMS-1,-1,-1)] # (ones ... K ... ones)
    kernels.append(Tensor.ones(K, *[1 for _ in range(i)]).reshape(line_shape).pad(padding)) # appply reshape and padding
    print(kernels[i].numpy())
