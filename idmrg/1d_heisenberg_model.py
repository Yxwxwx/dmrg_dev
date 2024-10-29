import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time

class HeisenbergDMRG:
    def __init__(self, L, J=1.0, max_states=50, convergence_threshold=1e-5):
        self.L = L  
        self.J = J  
        self.max_states = max_states  
        self.convergence_threshold = convergence_threshold  # 收敛阈值
        
        self.Sx = scipy.sparse.csr_matrix(0.5 * np.array([[0., 1.], [1., 0.]]))
        self.Sy = scipy.sparse.csr_matrix(0.5 * np.array([[0., -1j], [1j, 0.]]))
        self.Sz = scipy.sparse.csr_matrix(0.5 * np.array([[1., 0.], [0., -1.]]))
        self.I2 = scipy.sparse.csr_matrix(np.eye(2))  # 保持单位矩阵为稀疏格式
        
        self.left_block = self._init_site()
        self.right_block = self._init_site()
        self.current_size = 1  

    def _init_site(self):
        return {
            'H': scipy.sparse.csr_matrix((2, 2)),  # 保持稀疏格式
            'Sx': self.Sx,
            'Sy': self.Sy,
            'Sz': self.Sz
        }
    
    def enlarge_block(self, block):
        """
        扩大量子块的维度，包括自旋耦合的效应
        
        H' = H ⊗ I + J * (Sᵡ ⊗ Sᵡ + Sʏ ⊗ Sʏ + Sᶻ ⊗ Sᶻ)
        """
        dim = block['H'].shape[0]
        H_new = scipy.sparse.kron(block['H'], self.I2, format='csr')
        identity_dim = scipy.sparse.identity(dim, format='csr')

        H_new += self.J * (scipy.sparse.kron(block['Sx'], self.Sx, format='csr') +
                           scipy.sparse.kron(block['Sy'], self.Sy, format='csr') +
                           scipy.sparse.kron(block['Sz'], self.Sz, format='csr'))
        
        return {
            'H': H_new,
            'Sx': scipy.sparse.kron(identity_dim, self.Sx, format='csr'),
            'Sy': scipy.sparse.kron(identity_dim, self.Sy, format='csr'),
            'Sz': scipy.sparse.kron(identity_dim, self.Sz, format='csr')
        }
    
    def get_superblock_hamiltonian(self, left_block, right_block):
        """构造超级块哈密顿量
        H_super = H_left ⊗ I_right + I_left ⊗ H_right + J * (Sᵡ_left ⊗ Sᵡ_right + Sʏ_left ⊗ Sʏ_right + Sᶻ_left ⊗ Sᶻ_right)
        """
        left_dim = left_block['H'].shape[0]
        right_dim = right_block['H'].shape[0]
        
        H_super = (scipy.sparse.kron(left_block['H'], scipy.sparse.identity(right_dim, format='csr')) +
                    scipy.sparse.kron(scipy.sparse.identity(left_dim, format='csr'), right_block['H']))

        H_super += (self.J * (
            scipy.sparse.kron(left_block['Sx'], right_block['Sx']) +
            scipy.sparse.kron(left_block['Sy'], right_block['Sy']) +
            scipy.sparse.kron(left_block['Sz'], right_block['Sz'])
        ))
    
        return H_super
    
    def dmrg_step(self):
        """执行一次DMRG迭代
        1. 扩展左右量子块
        2. 构造超级块哈密顿量
        3. 求解基态能量和波函数
        4. 计算约化密度矩阵
        5. 对密度矩阵进行对角化以获取保留的态
    
        约化密度矩阵公式:
        ρ = |ψ⟩⟨ψ| / Tr(|ψ⟩⟨ψ|) 
        其中 |ψ⟩ 为超级块的基态波函数。
        """
        left_enlarged = self.enlarge_block(self.left_block)
        right_enlarged = self.enlarge_block(self.right_block)

        left_dim = left_enlarged['H'].shape[0]
        right_dim = right_enlarged['H'].shape[0]
        
        H_super = self.get_superblock_hamiltonian(left_enlarged, right_enlarged)
        energy, psi = scipy.sparse.linalg.eigsh(H_super, k=1, which='SA')
        psi = scipy.sparse.csr_matrix(psi.flatten().reshape(left_dim, right_dim))

        left_dim = left_enlarged['H'].shape[0]
        right_dim = right_enlarged['H'].shape[0]

        # 计算左右块的约化密度矩阵
        rho_left = psi @ psi.conj().T
        rho_left /= rho_left.trace()

        rho_right = psi.T @ psi.conj()
        rho_right /= rho_right.trace()
        
        U_left, s_left, _ = scipy.linalg.svd(rho_left.toarray(), full_matrices=False)
        U_right, s_right, _ = scipy.linalg.svd(rho_right.toarray(), full_matrices=False)

        num_states = min(self.max_states, len(s_left), len(s_right))

        self.left_block = {
            'H': self._transform_operator(left_enlarged['H'], U_left[:, :num_states]),
            'Sx': self._transform_operator(left_enlarged['Sx'], U_left[:, :num_states]),
            'Sy': self._transform_operator(left_enlarged['Sy'], U_left[:, :num_states]),
            'Sz': self._transform_operator(left_enlarged['Sz'], U_left[:, :num_states])
        }

        self.right_block = {
            'H': self._transform_operator(right_enlarged['H'], U_right[:, :num_states]),
            'Sx': self._transform_operator(right_enlarged['Sx'], U_right[:, :num_states]),
            'Sy': self._transform_operator(right_enlarged['Sy'], U_right[:, :num_states]),
            'Sz': self._transform_operator(right_enlarged['Sz'], U_right[:, :num_states])
        }

        self.current_size += 1
        return energy[0], np.sum(s_right[:num_states]) + np.sum(s_left[:num_states])

    def _transform_operator(self, operator, transformation_matrix):
        """对算符进行变换以得到新的算符
        
        A' = T† A T
        其中 T 为变换矩阵。
        """
        return transformation_matrix.conj().T @ operator @ transformation_matrix
    
    def run(self):
        energies = []
        truncation_errors = []
        
        for step in range(self.L - 1):
            energy, truncation_weight = self.dmrg_step()
            per_site_energy = energy / (self.current_size * 2)
            truncation_error = 2 - truncation_weight
            
            if energies and abs(per_site_energy - energies[-1]) < self.convergence_threshold:
                print(f"Converged at step {self.current_size}: Per site energy = {per_site_energy:.10f}")
                break
            
            energies.append(per_site_energy)
            truncation_errors.append(truncation_error)
            
            print(f"Step {self.current_size:2d}: "
                  f"Energy = {energy:.10f}, "
                  f"Per site energy = {per_site_energy:.10f}, "
                  f"Truncation Error = {truncation_error:.2e}")
        
        return energies, truncation_errors

def main():
    L = 100
    J = 1.0
    max_states = 20
    convergence_threshold = 1e-10  # 收敛阈值

    exact_energy_per_site = -np.log(2) + 0.25
    exact_total_energy = L * exact_energy_per_site
    
    dmrg = HeisenbergDMRG(L, J, max_states, convergence_threshold)
    energies, truncation_errors = dmrg.run()
    
    print(f"\nResults for {L}-site Heisenberg chain:")
    print(f"DMRG ground state energy: {energies[-1] * L:.10f}")
    print(f"Exact ground state energy: {exact_total_energy:.10f}")
    print(f"Relative error: {abs(energies[-1] * L - exact_total_energy) / abs(exact_total_energy):.2e}")
    print(f"Maximum truncation error: {max(truncation_errors):.2e}")
    
    print(f"\nPer site energies:")
    print(f"DMRG: {energies[-1]:.10f}")
    print(f"Exact: {exact_energy_per_site:.10f}")
    
    return energies, truncation_errors

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start:.2f} s")
