from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.sparse import linalg
from numpy.typing import NDArray


@dataclass
class MPO:
    """Matrix Product Operator (MPO) for representing Hamiltonians."""

    local_dim: int
    num_sites: int
    mpo: Optional[List[NDArray]] = None

    def __post_init__(self) -> None:
        """Initialize MPO after dataclass creation."""
        self.mpo = self.construct_mpo()

    def construct_mpo(self) -> List[NDArray]:
        """Constructs MPO for the Heisenberg Hamiltonian."""
        identity = np.identity(2)
        zeros = np.zeros((2, 2))
        sz = np.array([[0.5, 0], [0, -0.5]])
        sp = np.array([[0, 0], [1, 0]])
        sm = np.array([[0, 1], [0, 0]])

        w_bulk = np.array(
            [
                [identity, zeros, zeros, zeros, zeros],
                [sp, zeros, zeros, zeros, zeros],
                [sm, zeros, zeros, zeros, zeros],
                [sz, zeros, zeros, zeros, zeros],
                [zeros, 0.5 * sm, 0.5 * sp, sz, identity],
            ]
        )

        w_first = np.array([[zeros, 0.5 * sm, 0.5 * sp, sz, identity]])
        w_last = np.array([[identity], [sp], [sm], [sz], [zeros]])

        return [w_first] + [w_bulk] * (self.num_sites - 2) + [w_last]


@dataclass
class MPS:
    """Matrix Product State (MPS) representing the state of the system."""

    local_dim: int
    num_sites: int
    mps: Optional[List[NDArray]] = None

    def __post_init__(self) -> None:
        """Initialize MPS after dataclass creation."""
        self.mps = self.initialize_state()

    def initialize_state(self) -> List[NDArray]:
        """Initializes the MPS state to |01010101...>"""
        initial_a1 = np.zeros((self.local_dim, 1, 1))
        initial_a1[0, 0, 0] = 1
        initial_a2 = np.zeros((self.local_dim, 1, 1))
        initial_a2[1, 0, 0] = 1
        return [initial_a1, initial_a2] * (self.num_sites // 2)

    def truncate(
        self, u: NDArray, s: NDArray, v: NDArray, max_dim: int
    ) -> Tuple[NDArray, NDArray, NDArray, float, int]:
        """Truncates MPS tensors by retaining the max_dim largest singular values."""
        dim = min(len(s), max_dim)
        truncation = float(np.sum(s[dim:]))
        s = s[:dim]
        u = u[..., :dim]
        v = v[:, :dim, :]

        return u, s, v, truncation, dim


class HamiltonianOperator(linalg.LinearOperator):
    """Hamiltonian-vector multiplication operator for eigensolver."""

    def __init__(self, e: NDArray, w: NDArray, f: NDArray):
        self.e = e
        self.w = w
        self.f = f
        self.req_shape = [w.shape[2], e.shape[1], f.shape[1]]
        size = np.prod(self.req_shape)
        super().__init__(dtype=np.dtype("float64"), shape=(size, size))

    def _matvec(self, vector: NDArray) -> NDArray:
        """Implements matrix-vector product with Hamiltonian MPO representation."""
        shaped_vector = np.reshape(vector, self.req_shape)
        result = np.einsum("aij,sik->ajsk", self.e, shaped_vector, optimize=True)
        result = np.einsum("ajsk,abst->bjtk", result, self.w, optimize=True)
        result = np.einsum("bjtk,bkl->tjl", result, self.f, optimize=True)
        return np.reshape(result, -1)


class HeisenbergModel1Site:
    """Heisenberg model implementing the one-site DMRG algorithm."""

    def __init__(self, mps: MPS, mpo: MPO, bond_dim: int, num_sweeps: int):
        self.mps = mps
        self.mpo = mpo
        self.bond_dim = bond_dim
        self.num_sweeps = num_sweeps

    def optimize_one_site(
        self, site: int, e: NDArray, f: NDArray
    ) -> Tuple[float, NDArray, float, int]:
        """Optimizes a single MPS tensor at a given site."""
        hamiltonian = HamiltonianOperator(e, self.mpo.mpo[site], f)
        tensor = self.mps.mps[site]
        eigenvalue, eigenvector = linalg.eigsh(hamiltonian, k=1, v0=tensor, which="SA")
        tensor_reshaped = np.reshape(eigenvector[:, 0], tensor.shape)
        u, s, v = np.linalg.svd(tensor_reshaped, full_matrices=False)
        u, s, v, truncation, states = self.mps.truncate(u, s, v, self.bond_dim)
        return eigenvalue[0], u, truncation, states

    def run_dmrg(self, arg="debug") -> List[NDArray]:
        """Runs the one-site DMRG algorithm."""
        e_matrices, f_matrices = self.construct_boundaries()
        for sweep in range(self.num_sweeps):
            # Sweep over all sites
            for i in range(len(self.mps.mps)):
                energy, self.mps.mps[i], trunc, states = self.optimize_one_site(
                    i, e_matrices[-1], f_matrices[-1]
                )
                if arg != "quiet":
                    print(
                        f"Sweep {sweep} Site {i}    "
                        f"Energy {energy:16.12f}    States {states:4} "
                        f"Truncation {trunc:16.12f}"
                    )

        return self.mps.mps

    def construct_boundaries(self) -> Tuple[List[NDArray], List[NDArray]]:
        """Constructs boundary matrices for DMRG sweeps."""
        f_matrices = [np.array([[[1.0]]])]
        e_matrices = [np.array([[[1.0]]])]
        return e_matrices, f_matrices

    def calculate_expectation(self) -> float:
        """Calculates the expectation value of the Hamiltonian for the MPS."""
        expectation = np.array([[[1.0]]])
        for i in range(len(self.mpo.mpo)):
            expectation = np.einsum(
                "aij,abc,bkl->cjk",
                expectation,
                self.mpo.mpo[i],
                self.mps.mps[i],
                optimize=True,
            )
        return float(expectation[0][0][0])


def main() -> None:
    """Main function to demonstrate 1-site DMRG calculation."""
    LOCAL_DIM = 2
    NUM_SITES = 10
    BOND_DIM = 10
    NUM_SWEEPS = 2

    mpo = MPO(local_dim=LOCAL_DIM, num_sites=NUM_SITES)
    mps = MPS(local_dim=LOCAL_DIM, num_sites=NUM_SITES)
    model = HeisenbergModel1Site(
        mps=mps, mpo=mpo, bond_dim=BOND_DIM, num_sweeps=NUM_SWEEPS
    )

    try:
        model.run_dmrg("debug")
        final_energy = model.calculate_expectation()
        print("\nFinal Results:")
        print(f"Energy expectation value: {final_energy:16.12f}")

    except Exception as e:
        print(f"Error during DMRG calculation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
