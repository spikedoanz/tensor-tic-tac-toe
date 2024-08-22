from tinygrad import Tensor

from functools import reduce

DIMS = [2,2,2]
RANK = len(DIMS)
K = 4


# lines have rank permutations, one for each axis
# diagonals have rank*(rank-1) permutations, each extra axis creates a new type of diagonal, and each diagonal has rank permutations
# higher order identities are mutlivectors lmao

def add_axes(x: Tensor, rank: int, shift=0):
    shape = ' '.join(['1' if (i < shift or i > shift)  else '...' for i in range(rank-len(x.shape)+1)])
    return Tensor.rearrange(x, f"... -> {shape}")

def pad_axes(x: Tensor, shift=0):
    shifts = [(shift, max(x.shape)-i-shift) if i < max(x.shape) else (0,0) for i in x.shape]
    return x.pad(shifts)

def kronecker_delta(n, rank):
    def axis_vector(n, axis, rank): return add_axes(Tensor.ones(n), rank, axis)
    lines = [ [pad_axes(axis_vector(n, axis, rank), idx) for axis in range(rank)] for idx in range(n)] # create axis permuted lines at every index
    dots = [ reduce(lambda a, b: a*b, line) for line in lines]                                         # select out i==j==k==... for every index
    return reduce(lambda a,b: a.int()|b.int(), dots)                                                   # sum up the selected indeces

diag = kronecker_delta(2,2)

added_diag = add_axes(diag, 3,0)

padded_diag = pad_axes(added_diag,0)


print(diag.numpy())
print(added_diag.numpy())
print(padded_diag.numpy())
"""
rank = RANK
n = K
lines = [padded_line(n, axis, rank) for axis in range(rank)]
diags = []
for axis in range(1,rank+1):
    diag = identity(n, axis)

"""




"""
# tests
for i in range(1,100):
    assert (Tensor.ones(i,i).sum() == (Tensor.eye(i) == identity(i, 2)).sum()).numpy(), "{i}x{i} identity not working"
"""
