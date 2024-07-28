from tinygrad import Tensor
import numpy as np

N = 512
K = 64
N_BOARDS = 1

def convs(N=N, K=K):
    kernels = []
    for _ in range(1,K-1):
        line = np.zeros((K,K), dtype="int8")
        line[_, :] = 1
        kernels.append(line.reshape((1,1,K,K)))
        kernels.append(line.T.reshape((1,1,K,K)))

    diag = np.eye(K, dtype="int8").reshape((1,1,K,K))
    kernels.append(diag)
    kernels.append(np.flip(diag, axis=0))
    win_conv = Tensor(np.stack(kernels, axis=1)).squeeze(dim=0)
    return win_conv

def check(boards, kernels, N=N, K=K, specifics=True):
    n_boards = (K*2-2)*boards.shape[0]
    ret = boards.pad2d((1,1,1,1)).expand(n_boards,1,1+N+1,1+N+1).conv2d(kernels)
    one = Tensor.full(ret.shape, K)
    neg = Tensor.full(ret.shape, -K)
    one_match = (ret == one)
    neg_match = (ret == neg)
    o = one_match.sum()
    n = neg_match.sum()
    o_wins = o.item() / (K*2-2)
    n_wins = n.item() / (K*2-2)
    if specifics:
        print(f"Positions where one wins: {int(o_wins)}")
        print(f"Positions where -one wins: {int(n_wins)}")
    return (o_wins > 0)  - (n_wins > 0)

if __name__ == "__main__":
    print(f"Board size: {(N,N)}")
    print(f"Line length required for win: {K}")
    for _ in range(N_BOARDS):
        boards = Tensor.randint(1, N,N, low=-1, high=2)
        print(boards.numpy())
        kernels = convs()
        print(f"Actual winner: {check(boards, kernels)}")
