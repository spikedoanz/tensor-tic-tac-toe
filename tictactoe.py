from tinygrad import Tensor

from functools import reduce
from itertools import combinations

BOARD_DIM = [3 for _ in range(4)]
RANK = len(BOARD_DIM)
K = 3

def add_axes(x:Tensor, rank:int, shift:int=0):
    """ Adds extra axes around the tensor to turn into rank, shifted by 'shift'.' 
        Ex: x.shape == (3,3), add_axes(x, rank=5, shift=2) turns it into shape (1,1,3,3,1) """
    shape = ' '.join(['1' if (i < shift or i > shift)  else '...' for i in range(rank-len(x.shape)+1)])
    return Tensor.rearrange(x, f"... -> {shape}")

def pad_axes(x:Tensor, shift:int=0):
    """ Turns tensors of non hypercubic shape into hypercubic. Ex: 1 1 1 3 -> 3 3 3 3 """
    shifts = [(shift, max(x.shape)-i-shift) if i < max(x.shape) else (0,0) for i in x.shape]
    return x.pad(shifts)

def kronecker_delta(n:int, rank:int):
    """ https://en.wikipedia.org/wiki/Kronecker_delta """
    def axis_vector(n, axis, rank): return add_axes(Tensor.ones(n), rank, axis)
    lines = [ [pad_axes(axis_vector(n, axis, rank), idx) for axis in range(rank)] for idx in range(n)] # create axis permuted lines at every index
    dots = [ reduce(lambda a, b: a*b, line) for line in lines]                                         # select out i==j==k==... for every index
    return reduce(lambda a,b: a.int()|b.int(), dots)                                                   # sum up the selected indeces

def convs(K=K, RANK=RANK):
    kernels = []
    for local_rank in range(1,RANK+1):
        kernel = pad_axes(add_axes(kronecker_delta(K,local_rank), RANK))
        flips = [kernel.flip(r) for r in range(local_rank)]
        kernels.extend(flips)
        if local_rank != RANK: 
            transpositions = list(combinations(range(RANK), 2))
            print(transpositions)
            kernels.extend([kernel.transpose(*t) for t in transpositions for kernel in flips])
    ret = [Tensor.rearrange(kernel, '... -> 1 ...') for kernel in kernels]
    return Tensor.stack(*ret, dim=0)

def check(board:Tensor, kernels:Tensor, specifics:bool=False):
    padded_board = board.pad( [(0, K) for _ in board.shape])
    ret = Tensor.rearrange(padded_board, '... -> 1 1 ...').conv2d(kernels)
    one, neg = [Tensor.full(ret.shape, i) for i in (K, -K)]
    o_wins, x_wins = [(ret == player).sum().item() / (K*2-2) for player in (one, neg)]
    if specifics:
        print(f"Positions where 1 wins : {int(o_wins)}")
        print(f"Positions where -1 wins: {int(x_wins)}")
    return (o_wins > 0) - (x_wins > 0)

if __name__ == "__main__":
    board = Tensor.randint(*BOARD_DIM, low=-1, high=2)
    kernels = convs()
    print(kernels.numpy())
    result = check(board, kernels, True)
    print(board.numpy())
    print(BOARD_DIM)
