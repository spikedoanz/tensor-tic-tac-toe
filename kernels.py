from tinygrad import Tensor

DIMS = [2,2]
NDIM = len(DIMS)
K = 2

kernels = []
# line kernels (can probably turn this into a monstrous single line list comprehension)
# simplify using flip
def line_kernel(n, K=K, rank=NDIM): # create a rank NDIMS 
    padding = tuple([(0,K-1) if i != n else (0,0) for i in range(rank-1,-1,-1)])   # ((pad to K) ... don't pad ... (pad to K))
    line_shape = [K if i == n else 1 for i in range(rank-1,-1,-1)]                 # (ones ... K ... ones)
    line = Tensor.ones(K, *[1 for _ in range(n)])                                   # 1s along i'th axis
    return line.reshape(line_shape).pad(padding)                                    # appply reshape and padding to get to rank NDIMS all Ks

def line_at(n, K=K, rank=NDIM):
    line = Tensor.ones(K)
    # pad 0s around N to get to K
    # pad KxK 0s around KxK to get to rank
    # flip to some axis
    return line

# create the identity
    # create NDIM line tensors x K positions
    # sum([[reduce(lines_at_same_axis, ==) for lines in all lines_at(n)] for n in range(K)])
    # => classic i=j=k... identity
    # NDIM flips to create all variants


# stack all flipped lines and eyes into a single NDIM-conv
# define board (easy)
# do the same trick as in 2d tictactpe variant
