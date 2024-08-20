from tinygrad import Tensor
import numpy as np

BOARD_DIM = [4,3]
K = 3
N_BOARDS = 1

def convs(K=K):
    kernels = []
    kernel_dim = tuple([K for _ in BOARD_DIM])
    for _ in range(1,K-1):
        line = np.zeros(kernel_dim, dtype="int8")
        line[_, :] = 1
        kernels.append(line.reshape((1,1,K,K)))
        kernels.append(line.T.reshape((1,1,K,K)))

    diag = np.eye(K, dtype="int8").reshape((1,1,K,K))
    kernels.append(diag)
    kernels.append(np.flip(diag, axis=0))
    win_conv = Tensor(np.stack(kernels, axis=1)).squeeze(dim=0)
    return win_conv

def check(board, kernels,specifics=True):
    ret = Tensor.rearrange(board.pad2d((1,1,1,1)), '... -> 1 1 ...').conv2d(kernels)
    one = Tensor.full(ret.shape, K)
    neg = Tensor.full(ret.shape, -K)
    o_wins = (ret == one).sum().item() / (K*2-2)
    n_wins = (ret == neg).sum().item() / (K*2-2)
    if specifics:
        print(f"Positions where one wins: {int(o_wins)}")
        print(f"Positions where -one wins: {int(n_wins)}")
    return (o_wins > 0)  - (n_wins > 0)

if __name__ == "__main__":
    print(f"Board size: {BOARD_DIM}")
    print(f"Line length required for win: {K}")
    for _ in range(N_BOARDS):
        board = Tensor.randint(*BOARD_DIM, low=-1, high=2)
        kernels = convs()
        result = check(board, kernels)
        print(board.numpy())
        print(f"Actual winner: {result}")
