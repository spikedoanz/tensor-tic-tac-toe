# Tensor Tic Tac Toe

Parallelized and generalized implementation of a hyperdimensional TicTacToe in pure tensor ops

![nazuna](dingcube.png)

--- 

## Spec

BOARD_DIM: Tuple[int] = Dimensions of the board. Can be N-dimensional (ex: [50,50,50,50,50])

K: int = Length of line required to win

## Usage

```bash
git clone https://github.com/spikedoanz/tensor-tic-tac-toe
cd tensor-tic-tac-toe
git clone git@github.com:tinygrad/tinygrad.git tinygrad-nightly
pip install -e tinygrad-nightly
python hyperdimensional_tictactoe.py
```

## How this works

The tic tac toe of shape BOARD_DIM encodes for moves with 1 or -1

Win conditions of tic tac toe can be encoded by using a convolution.

The kernel of said convolution lets us select out the positions we like (lines, diagonals, etc)

Example: for a (3,3) board, that requires 3 things in a row to win:

```
[[[[0 0 0] 
   [1 1 1]
   [0 0 0]]] # horizontal

 [[[0 1 0]
   [0 1 0]
   [0 1 0]]] # vertical

 [[[1 0 0]
   [0 1 0]
   [0 0 1]]] # diagonal

 [[[0 0 1]
   [0 1 0]
   [1 0 0]]]] # anti-diagonal
```

The board is then padded to allow the conv to go over the edges of the board

Kernels can then be applied as a convolution (basically a bitmap) over the board
```
[1  1 -1  0  0]
[1  0  0  0  0]             [1 0 0]     [3   2  -1] 
[1  1  0  0  0]  .conv2d()  [1 0 0] ->  [2   1   0]
[0  0  0  0  0]             [1 0 0]     [1   1   0]
[0  0  0  0  0]
```

Then, an element wise equality with a tensor full of k or -k is applied to the resultant tensor
```
[3   2  -1]      [3  3  3]       [T  F  F]
[2   1   0]  ==  [3  3  3]  ->   [F  F  F]
[1   1   0]      [3  3  3]       [F  F  F]
```

The tensor is then summed, and if it's bigger than 0, a win is found! 
```
[T  F  F]
[F  F  F]  .sum()  -> 1 > 0 ==> Win!!!
[F  F  F]
```

Win conditions are fully generalized thanks to the [kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta), which is a tensor with 1s only where i=j=k=...
```
def kronecker_delta(n:int, rank:int):
    def axis_vector(n, axis, rank): return add_axes(Tensor.ones(n), rank, axis) # creates a row or column or ... vector of ones (line)
    lines = [ [pad_axes(axis_vector(n, axis, rank), idx) for axis in range(rank)] for idx in range(n)] # create axis permuted lines at every index
    dots = [ reduce(lambda a, b: a*b, line) for line in lines]                                         # select out i==j==k==... for every index
    return reduce(lambda a,b: a.int()|b.int(), dots)                                                   # sum up the selected indeces
```


Each rank has its own type of win condition:

Rank 1: Lines

```
            [1 1 1]
```

Rank 2: Diagonals of a square
```
[
            [1 0 0]
            [0 1 0]
            [0 0 1]
]
```


Rank 3: Diagonals of a cube
```
[
    [
              [[1 0 0]      [[0 0 0]      [[0 0 0]
               [0 0 0]       [0 1 0]       [0 0 0]
               [0 0 0]]      [0 0 0]       [0 0 1]]
    ]
]
```


Rank 4: Diagonals of a hypercube
```
RANK = 4
print(convs(RANK=4)[-1].numpy())
```

Rank N: Diagonals of an N-cube
```
RANK = N
print(convs(RANK=4)[-1].numpy())
```
