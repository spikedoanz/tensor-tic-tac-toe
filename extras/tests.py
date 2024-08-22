from tinygrad import Tensor

from tictactoe import convs, check

from hyperdimensional_tictactoe import kronecker_delta
from hyperdimensional_tictactoe import convs as hyperconvs
from hyperdimensional_tictactoe import check as hypercheck 

def test_kronecker():
    for i in range(1,100):
        assert (Tensor.ones(i,i).sum() == (Tensor.eye(i) == kronecker_delta(i, 2)).sum()).numpy(), "2D kronecker delta fails"

def test_2d_tic_tac_toe():
    N = 512
    K = 32
    N_BOARDS = 10
    for _ in range(N_BOARDS): 
        board = Tensor.randint(N,N, low=-1, high=2)
        kernels = convs(K)
        hyper_kernels = hyperconvs(K, RANK=2)
        assert hypercheck(board, hyper_kernels) == check(board, kernels), "spec is not met"


if __name__ == "__main__":
    test_2d_tic_tac_toe()

