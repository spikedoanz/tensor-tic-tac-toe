# Tensor Tic Tac Toe

Fully parallel implementation of TicTacToe win condition verifier

--- 

## Spec

n: dimension of board (n,n) 
> ( might do 3d for fun lol ) 
>
> ( nd would require fully generalized conv kernel, which i don't have)

k: length of line required to win

## How this works

Win conditions are factored into 4 types: horizontal, vertical, diagonal and anti-diagonal

Vertical is the transpose (flip also works) of horizontal, and anti-diagonal is the flip of diagonal









--- 

## Text I sent to a friend in a haze at 5am

> just realized i can formulate the win conditions of tictactoe with a convolution

> the kernels are the horizontal and verticals of 1..n-1 indexes filled out with 1
>   plus identity and flipped identity for the diagonals
> so a board of size n and line win condition k can be verified in parallel as a
> conv, pad 1, stride 1
> with a (k,n,n) kernel (k-2 for horizontal/vertical, 2 for diagonals)

> kernels are then pooled, then checked for occurrences of k. (maybe with a flattened tensor of shape (2, k,n,n) with ks and -ks on every entry. then summed. 1 means win, 0 means loss
