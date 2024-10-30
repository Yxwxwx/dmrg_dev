import numpy as np
import scipy.sparse.linalg
import time
from typing import List, Tuple

class HeisenbergMPS:
    def __init__(self, L: int, J: float = 1.0, max_bond_dim: int = 50, convergence_threshold: float = 1e-5):
        self.L = L
        self.J = J
        self.max_bond_dim = max_bond_dim
        self.convergence_threshold = convergence_threshold
        
        # Pauli matrices
        self.Sx = 0.5 * np.array([[0., 1.], [1., 0.]])
        self.Sy = 0.5 * np.array([[0., -1j], [1j, 0.]])
        self.Sz = 0.5 * np.array([[1., 0.], [0., -1.]])
        self.I2 = np.eye(2)
        
        # Initialize MPS and MPO
        self.mps = self._init_mps()
        self.mpo = self._init_mpo()
        
    def _init_mps(self) -> List[np.ndarray]:
        """Initialize random MPS"""
        mps = []
        D = min(2, self.max_bond_dim)
        
        # First site: (1 x 2 x D)
        site = np.random.random((1, 2, D)) + 1j * np.random.random((1, 2, D))
        mps.append(site / np.linalg.norm(site))
        
        # Middle sites: (D x 2 x D)
        for _ in range(1, self.L-1):
            site = np.random.random((D, 2, D)) + 1j * np.random.random((D, 2, D))
            mps.append(site / np.linalg.norm(site))
            
        # Last site: (D x 2 x 1)
        site = np.random.random((D, 2, 1)) + 1j * np.random.random((D, 2, 1))
        mps.append(site / np.linalg.norm(site))
        
        return mps
    
    def _init_mpo(self) -> List[np.ndarray]:
        """Initialize Heisenberg MPO"""
        mpo = []
        
        # First site: (1 x 5 x 2 x 2)
        W = np.zeros((1, 5, 2, 2), dtype=complex)
        W[0, 0] = self.I2
        W[0, 1] = self.J * self.Sx
        W[0, 2] = self.J * self.Sy
        W[0, 3] = self.J * self.Sz
        W[0, 4] = -self.I2
        mpo.append(W)
        
        # Middle sites: (5 x 5 x 2 x 2)
        W = np.zeros((5, 5, 2, 2), dtype=complex)
        W[0, 0] = self.I2
        W[1, 4] = self.Sx
        W[2, 4] = self.Sy
        W[3, 4] = self.Sz
        W[4, 4] = self.I2
        
        for _ in range(self.L-2):
            mpo.append(W.copy())
        
        # Last site: (5 x 1 x 2 x 2)
        W = np.zeros((5, 1, 2, 2), dtype=complex)
        W[4, 0] = self.I2
        mpo.append(W)
        
        return mpo
    
    def _contract_two_sites(self, site1: np.ndarray, site2: np.ndarray) -> np.ndarray:
        """Contract two adjacent MPS sites"""
        return np.tensordot(site1, site2, axes=(2, 0))
    
    def _contract_mps_mpo_two_sites(self, two_site_state: np.ndarray, 
                                     mpo1: np.ndarray, mpo2: np.ndarray) -> np.ndarray:
        """Apply two MPO sites to a two-site state"""
        # Contract first site with first MPO
        temp1 = np.tensordot(two_site_state, mpo1, axes=([1, 2], [2, 3]))  # (1, 5, 2)
        
        # Contract second site with second MPO
        temp2 = np.tensordot(temp1, mpo2, axes=([1, 2], [2, 3]))  # (1, 2)
    
        return temp2

    def _svd_truncate(self, matrix: np.ndarray, max_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Perform SVD decomposition with truncation"""
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        
        truncation_dim = min(len(S), max_dim)
        truncation_error = 1 - np.sum(S[:truncation_dim]**2) / np.sum(S**2)
        
        U = U[:, :truncation_dim]
        S = S[:truncation_dim]
        Vh = Vh[:truncation_dim, :]

        return U, S, Vh, truncation_error
    
    def sweep(self, direction: str = 'right') -> Tuple[float, float]:
        """Execute one DMRG sweep"""
        if direction not in ['right', 'left']:
            raise ValueError("direction must be 'right' or 'left'")

        sites = range(self.L-1) if direction == 'right' else range(self.L-1, 0, -1)
        max_truncation_error = 0
        energy = 0

        for i in sites:
            # Contract two sites
            if direction == 'right':
                two_site = self._contract_two_sites(self.mps[i], self.mps[i+1])
                H_two_site = self._contract_mps_mpo_two_sites(two_site, self.mpo[i], self.mpo[i+1])
            else:
                two_site = self._contract_two_sites(self.mps[i-1], self.mps[i])
                H_two_site = self._contract_mps_mpo_two_sites(two_site, self.mpo[i-1], self.mpo[i])

            # Reshape for eigenvalue problem
            a1, s1, a2, s2 = two_site.shape
            left_dim = a1 * s1
            right_dim = a2 * s2
            
            print(f"two_site shape: {two_site.shape}")
            print(f"H_two_site shape before reshape: {H_two_site.shape}")

            # Reshape state and Hamiltonian consistently
            state_vector = two_site.reshape(left_dim * right_dim)
            H_matrix = H_two_site.reshape(left_dim * right_dim, left_dim * right_dim)

            # Ensure matrix is Hermitian
            H_matrix = 0.5 * (H_matrix + H_matrix.conj().T)

            # Solve eigenvalue problem
            e, v = scipy.sparse.linalg.eigsh(H_matrix, k=1, which='SA', v0=state_vector)
            energy = e[0].real

            # Reshape eigenvector and perform SVD
            v = v[:, 0].reshape(a1 * s1, a2 * s2)
            U, S, Vh, trunc_err = self._svd_truncate(v, self.max_bond_dim)

            # Update MPS tensors
            if direction == 'right':
                self.mps[i] = U.reshape(a1, s1, -1)
                self.mps[i+1] = (np.diag(S) @ Vh).reshape(-1, s2, a2)
            else:
                self.mps[i-1] = U.reshape(a1, s1, -1)
                self.mps[i] = (np.diag(S) @ Vh).reshape(-1, s2, a2)

            max_truncation_error = max(max_truncation_error, trunc_err)

        return energy, max_truncation_error

    def run(self, max_sweeps: int = 20) -> Tuple[List[float], List[float]]:
        """Run DMRG optimization"""
        energies = []
        truncation_errors = []
        
        for sweep in range(max_sweeps):
            # Right sweep
            energy_right, trunc_err_right = self.sweep('right')
            # Left sweep
            energy_left, trunc_err_left = self.sweep('left')
            
            energy = (energy_right + energy_left) / 2
            trunc_err = max(trunc_err_right, trunc_err_left)
            
            energies.append(energy)
            truncation_errors.append(trunc_err)
            
            # Check convergence
            if sweep > 0 and abs(energies[-1] - energies[-2]) < self.convergence_threshold:
                print(f"Converged at sweep {sweep+1}")
                break
                
            print(f"Sweep {sweep+1}: Energy = {energy:.10f}, "
                  f"Truncation Error = {trunc_err:.2e}")
        
        return energies, truncation_errors

def main():
    # Model parameters
    L = 100
    J = 1.0
    max_bond_dim = 20
    convergence_threshold = 1e-10
    
    # Exact solution (Bethe ansatz)
    exact_energy_per_site = -np.log(2) + 0.25
    exact_total_energy = L * exact_energy_per_site
    
    # Run MPS simulation
    model = HeisenbergMPS(L, J, max_bond_dim, convergence_threshold)
    energies, truncation_errors = model.run()
    
    # Output results
    print(f"\nResults for {L}-site Heisenberg chain:")
    print(f"MPS ground state energy: {energies[-1]:.10f}")
    print(f"Exact ground state energy: {exact_total_energy:.10f}")
    print(f"Relative error: {abs(energies[-1] - exact_total_energy) / abs(exact_total_energy):.2e}")
    print(f"Maximum truncation error: {max(truncation_errors):.2e}")
    
    return energies, truncation_errors

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start:.2f} s")