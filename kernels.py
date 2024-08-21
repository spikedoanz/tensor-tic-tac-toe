from tinygrad import Tensor

from functools import reduce

DIMS = [2,2,2]
RANK = len(DIMS)
K = 2


# lines have rank permutations, one for each axis
# diagonals have rank*(rank-1) permutations, each extra axis creates a new type of diagonal, and each diagonal has rank permutations
# higher order identities are mutlivectors lmao

def line(n, axis, rank): return Tensor.rearrange(Tensor.ones(n), f"a -> {' '.join(['1' if i != axis else 'a' for i in range(rank)])}")
def padded_line(n, axis, rank, shift=0): return line(n, axis,rank).pad([(shift,n-1-shift) if i != axis else (0,0) for i in range(rank)])
def identity(n, rank):
    lines = [ [padded_line(n, axis, rank, sft) for axis in range(rank)] for sft in range(n)] # create axis permuted lines at every index
    dots = [ reduce(lambda a, b: a*b, line) for line in lines] # select out i==j==k==...
    return reduce(lambda a,b: a+b, dots) # sum up the selected indeces
    

rank = RANK
n = K
lines = [padded_line(n, axis, rank) for axis in range(rank)]
diags = []
for axis in range(2,rank+1-1):
    diag = identity(n, axis)
    padded= diag.pad( [(0, rank-2) if i > 2 else (0,0) for i in range(rank-axis+2,0,-1)])
    flips = [diag.flip(i) for i in range(rank-1)]
    print(flips[1].numpy())




"""
# tests
for i in range(1,100):
    assert (Tensor.ones(i,i).sum() == (Tensor.eye(i) == identity(i, 2)).sum()).numpy(), "{i}x{i} identity not working"
"""
