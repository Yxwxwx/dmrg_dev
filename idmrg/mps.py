import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
from typing import Tuple, List, Dict

class MatrixProductOperator:
    """
    矩阵积算符 (MPO) 的实现
    对于Heisenberg模型：H = J * ∑[1/2(S⁺ᵢS⁻ᵢ₊₁ + S⁻ᵢS⁺ᵢ₊₁) + SᶻᵢSᶻᵢ₊₁]
    """
    def __init__(self, L: int, J: float = 1.0):
        self.L = L  # 系统的物理位点数
        self.J = J  # 耦合常数
        
        # 定义算符
        self.I = scipy.sparse.csr_matrix(np.eye(2))
        self.Sz = scipy.sparse.csr_matrix(0.5 * np.array([[1., 0.], [0., -1.]]))
        self.Splus = scipy.sparse.csr_matrix(np.array([[0., 1.], [0., 0.]]))
        self.Sminus = scipy.sparse.csr_matrix(np.array([[0., 0.], [1., 0.]]))
        self.Zero = scipy.sparse.csr_matrix(np.zeros((2, 2)))
        
        # 初始化 MPO
        self.mpo = self._build_mpo()
        
    def _build_mpo(self):
        """构建 Heisenberg 模型的 MPO"""
        mpo = []
        
        # 第一个位点的 MPO 矩阵
        W1 = np.zeros((1, 5, 2, 2))
        W1[0, 0] = self.I.toarray()
        W1[0, 1] = self.Splus.toarray()
        W1[0, 2] = self.Sminus.toarray()
        W1[0, 3] = self.Sz.toarray()
        W1[0, 4] = self.Zero.toarray()
        mpo.append(W1)
        
        # 中间位点的 MPO 矩阵
        WM = np.zeros((5, 5, 2, 2))
        # 第一行
        WM[0, 0] = self.I.toarray()
        # 第二到四行
        WM[1, 0] = self.Splus.toarray()
        WM[2, 0] = self.Sminus.toarray()
        WM[3, 0] = self.Sz.toarray()
        # 最后一行
        WM[4, 1] = (self.J/2) * self.Sminus.toarray()
        WM[4, 2] = (self.J/2) * self.Splus.toarray()
        WM[4, 3] = self.J * self.Sz.toarray()
        WM[4, 4] = self.I.toarray()
        
        # 添加所有中间位点
        for i in range(1, self.L-1):
            mpo.append(WM.copy())
        
        # 最后一个位点的 MPO 矩阵
        WN = np.zeros((5, 1, 2, 2))
        WN[0, 0] = self.I.toarray()
        WN[1, 0] = (self.J/2) * self.Sminus.toarray()
        WN[2, 0] = (self.J/2) * self.Splus.toarray()
        WN[3, 0] = self.J * self.Sz.toarray()
        WN[4, 0] = self.I.toarray()
        mpo.append(WN)
        
        return mpo
    
    def get_mpo_dimensions(self):
        """获取 MPO 在每个位点的维度"""
        dimensions = []
        for tensor in self.mpo:
            dimensions.append(tensor.shape)
        return dimensions


class MatrixProductState:
    """矩阵积态 (MPS) 的实现"""
    def __init__(self, L: int, max_states: int = 50, linkdims: List[int] = None):
        self.L = L
        self.max_states = max_states
        
        # 处理 linkdims 输入
        if isinstance(linkdims, int):
            self.linkdims = [min(linkdims, max_states) for _ in range(L)]
        elif isinstance(linkdims, list):
            if len(linkdims) != L:
                raise ValueError(f"linkdims list must have the same length as L ({L}).")
            self.linkdims = [min(ld, max_states) for ld in linkdims]
        else:
            raise ValueError("linkdims must be an int or a list of ints.")
        
        self.tensors = self._initialize_mps()
    
    def _initialize_mps(self):
        """初始化随机的 MPS"""
        mps = []
        # 第一个位点
        tensor = np.random.rand(1, 2, self.linkdims[0])
        tensor /= np.linalg.norm(tensor)
        mps.append(tensor)
        
        # 中间位点
        for i in range(1, self.L - 1):
            tensor = np.random.rand(self.linkdims[i-1], 2, self.linkdims[i])
            tensor /= np.linalg.norm(tensor)
            mps.append(tensor)
        
        # 最后一个位点
        tensor = np.random.rand(self.linkdims[-2], 2, 1)
        tensor /= np.linalg.norm(tensor)
        mps.append(tensor)
        
        return mps



class HeisenbergDMRG:
    """Heisenberg DMRG 实现"""
    def __init__(self, L: int, J: float = 1.0, max_states: int = 50, convergence_threshold: float = 1e-5, linkdims: List[int] = None):
        self.L = L
        self.J = J
        self.max_states = max_states
        self.convergence_threshold = convergence_threshold
        
        # 初始化 MPS 和 MPO
        self.mps = MatrixProductState(L, max_states, linkdims)  # 传递 linkdims
        self.mpo = MatrixProductOperator(L, J)

def test_dmrg():
    L = 10
    # 示例 1: 使用相同的链接维度
    dmrg1 = HeisenbergDMRG(L, J=1.0, max_states=20, convergence_threshold=1e-5, linkdims=15)  # 所有链接维度均为 8
    
    # 打印维度信息
    print("MPS dimensions (uniform linkdims=15):", [tensor.shape for tensor in dmrg1.mps.tensors])
    print("MPO dimensions:", dmrg1.mpo.get_mpo_dimensions())

    
if __name__ == "__main__":
    test_dmrg()
    
