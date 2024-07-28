# Tensor Tic Tac Toe

Fully parallel implementation of TicTacToe win condition verifier

--- 

## Spec

n: dimension of board (n,n) 
> ( might do 3d for fun lol ) 
> ( nd would require fully generalized conv kernel, which i don't have)

k: length of line required to win

--- 

> just realized i can formulate the win conditions of tictactoe with a convolution, and a pool

> the kernels are the horizontal and verticals of 1..n-1 indexes filled out with 1
> plus identity and transposed identity for the diagonals
> so a board of size n and line win condition k can be verified in parallel as a
> conv, pad 1, stride 1
> with a (k,n,n) kernel (k-2 for horizontal/vertical, 2 for diagonals)

> kernels are then pooled, then checked for occurrences of k. (maybe with a flattened tensor of shape (2, k,n,n) with ks and -ks on every entry. then summed. 1 means win, 0 means loss
