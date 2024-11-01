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
        # Define local operators
        identity = np.identity(2)
        zeros = np.zeros((2, 2))
        sz = np.array([[0.5, 0], [0, -0.5]])
        sp = np.array([[0, 0], [1, 0]])  # S^+
        sm = np.array([[0, 1], [0, 0]])  # S^-

        # MPO tensors for each site in the chain
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

    def product(self, other: "MPO") -> List[NDArray]:
        """Multiplies two MPOs element-wise."""
        if not self.mpo or not other.mpo or len(self.mpo) != len(other.mpo):
            raise ValueError("Invalid MPO multiplication")

        return [self._tensor_product(w1, w2) for w1, w2 in zip(self.mpo, other.mpo)]

    @staticmethod
    def _tensor_product(w1: NDArray, w2: NDArray) -> NDArray:
        """Helper function to combine two MPO tensors."""
        product = np.einsum("abst,cdtu->acbdsu", w1, w2, optimize=True)
        new_shape = [
            w1.shape[0] * w2.shape[0],
            w1.shape[1] * w2.shape[1],
            w1.shape[2],
            w2.shape[3],
        ]
        return np.reshape(product, new_shape)

    def coarse_grain(self, w: NDArray, x: NDArray) -> NDArray:
        """Coarse-grains two site MPO tensors into a single tensor."""
        product = np.einsum("abst,bcuv->acsutv", w, x, optimize=True)
        return np.reshape(
            product,
            [w.shape[0], x.shape[1], w.shape[2] * x.shape[2], w.shape[3] * x.shape[3]],
        )


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

    @staticmethod
    def initial_boundary(w: NDArray, is_left: bool = True) -> NDArray:
        """Initializes boundary matrices for the vacuum state."""
        shape = w.shape[0] if is_left else w.shape[1]
        boundary = np.zeros((shape, 1, 1))
        boundary[0 if is_left else -1] = 1
        return boundary

    def coarse_grain(self, a: NDArray, b: NDArray) -> NDArray:
        """Coarse-grains two MPS sites into one."""
        product = np.einsum("sij,tjk->stik", a, b, optimize=True)
        return np.reshape(product, [a.shape[0] * b.shape[0], a.shape[1], b.shape[2]])

    def fine_grain(
        self, a: NDArray, dims: Tuple[int, int]
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Performs fine-graining on MPS, splitting a single coarse-grained site."""
        if a.shape[0] != dims[0] * dims[1]:
            raise ValueError("Invalid dimensions for fine graining")

        theta = np.transpose(
            np.reshape(a, dims + [a.shape[1], a.shape[2]]), (0, 2, 1, 3)
        )
        m_matrix = np.reshape(theta, (dims[0] * a.shape[1], dims[1] * a.shape[2]))
        u, s, v = np.linalg.svd(m_matrix, full_matrices=False)

        u = np.reshape(u, (dims[0], a.shape[1], -1))
        v = np.transpose(np.reshape(v, (-1, dims[1], a.shape[2])), (1, 0, 2))

        return u, s, v

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


@dataclass
class HeisenbergModel:
    """Heisenberg model implementing the two-site DMRG algorithm."""

    mps: MPS
    mpo: MPO
    bond_dim: int
    num_sweeps: int

    def contract_left(self, w: NDArray, a: NDArray, e: NDArray, b: NDArray) -> NDArray:
        """Contracts tensors from the left side."""
        temp = np.einsum("sij,aik->sajk", a, e, optimize=True)
        temp = np.einsum("sajk,abst->tbjk", temp, w, optimize=True)
        return np.einsum("tbjk,tkl->bjl", temp, b, optimize=True)

    def contract_right(self, w: NDArray, a: NDArray, f: NDArray, b: NDArray) -> NDArray:
        """Contracts tensors from the right side."""
        temp = np.einsum("sij,bjl->sbil", a, f, optimize=True)
        temp = np.einsum("sbil,abst->tail", temp, w, optimize=True)
        return np.einsum("tail,tkl->aik", temp, b, optimize=True)

    def construct_boundaries(self) -> Tuple[List[NDArray], List[NDArray]]:
        """Constructs boundary matrices for DMRG sweeps."""
        if not self.mpo.mpo or not self.mps.mps:
            raise ValueError("MPO or MPS not initialized")

        # Initialize boundaries
        f_matrices = [self.mps.initial_boundary(self.mpo.mpo[-1], is_left=False)]
        e_matrices = [self.mps.initial_boundary(self.mpo.mpo[0], is_left=True)]

        # Construct right boundary matrices
        for i in range(len(self.mpo.mpo) - 1, 0, -1):
            f_matrices.append(
                self.contract_right(
                    self.mpo.mpo[i], self.mps.mps[i], f_matrices[-1], self.mps.mps[i]
                )
            )

        return e_matrices, f_matrices

    def optimize_two_sites(
        self,
        a: NDArray,
        b: NDArray,
        w1: NDArray,
        w2: NDArray,
        e: NDArray,
        f: NDArray,
        direction: str,
    ) -> Tuple[float, NDArray, NDArray, float, int]:
        """Optimizes two-site tensors to minimize energy."""
        w = self.mpo.coarse_grain(w1, w2)
        aa = self.mps.coarse_grain(a, b)

        hamiltonian = HamiltonianOperator(e, w, f)
        eigenvalue, eigenvector = linalg.eigsh(hamiltonian, k=1, v0=aa, which="SA")

        aa = np.reshape(eigenvector[:, 0], hamiltonian.req_shape)
        a, s, b = self.mps.fine_grain(aa, [a.shape[0], b.shape[0]])
        a, s, b, truncation, states = self.mps.truncate(a, s, b, self.bond_dim)

        if direction == "right":
            b = np.einsum("ij,sjk->sik", np.diag(s), b, optimize=True)
        else:
            a = np.einsum("sij,jk->sik", a, np.diag(s), optimize=True)

        return eigenvalue[0], a, b, truncation, states

    def run_dmrg(self) -> List[NDArray]:
        """Runs the two-site DMRG algorithm."""
        e_matrices, f_matrices = self.construct_boundaries()
        f_matrices.pop()  # Remove last F matrix as it's not needed initially

        for sweep in range(self.num_sweeps // 2):
            # Right sweep
            for i in range(0, len(self.mps.mps) - 2):
                energy, self.mps.mps[i], self.mps.mps[i + 1], trunc, states = (
                    self.optimize_two_sites(
                        self.mps.mps[i],
                        self.mps.mps[i + 1],
                        self.mpo.mpo[i],
                        self.mpo.mpo[i + 1],
                        e_matrices[-1],
                        f_matrices[-1],
                        "right",
                    )
                )
                print(
                    f"Sweep {sweep*2} Sites {i},{i+1}    "
                    f"Energy {energy:16.12f}    States {states:4} "
                    f"Truncation {trunc:16.12f}"
                )

                e_matrices.append(
                    self.contract_left(
                        self.mpo.mpo[i],
                        self.mps.mps[i],
                        e_matrices[-1],
                        self.mps.mps[i],
                    )
                )
                f_matrices.pop()

            # Left sweep
            for i in range(len(self.mps.mps) - 2, 0, -1):
                energy, self.mps.mps[i], self.mps.mps[i + 1], trunc, states = (
                    self.optimize_two_sites(
                        self.mps.mps[i],
                        self.mps.mps[i + 1],
                        self.mpo.mpo[i],
                        self.mpo.mpo[i + 1],
                        e_matrices[-1],
                        f_matrices[-1],
                        "left",
                    )
                )
                print(
                    f"Sweep {sweep*2 + 1} Sites {i},{i+1}    "
                    f"Energy {energy:16.12f}    States {states:4} "
                    f"Truncation {trunc:16.12f}"
                )

                f_matrices.append(
                    self.contract_right(
                        self.mpo.mpo[i + 1],
                        self.mps.mps[i + 1],
                        f_matrices[-1],
                        self.mps.mps[i + 1],
                    )
                )
                e_matrices.pop()

        return self.mps.mps

    def calculate_expectation(self) -> float:
        """Calculates the expectation value of the Hamiltonian for the MPS."""
        if not self.mpo.mpo or not self.mps.mps:
            raise ValueError("MPO or MPS not initialized")

        expectation = np.array([[[1.0]]])
        for i in range(len(self.mpo.mpo)):
            expectation = self.contract_left(
                self.mpo.mpo[i], self.mps.mps[i], expectation, self.mps.mps[i]
            )
        return float(expectation[0][0][0])

    def calculate_variance(self) -> float:
        """Calculates the variance to measure the accuracy of the ground state energy."""
        if not self.mpo.mpo or not self.mps.mps:
            raise ValueError("MPO or MPS not initialized")

        ham_squared = self.mpo.product(self.mpo)  # Square of the Hamiltonian operator
        energy = self.calculate_expectation()
        h2_expectation = self._calculate_mpo_expectation(ham_squared)

        return float(h2_expectation - energy**2)

    def _calculate_mpo_expectation(self, mpo_tensors: List[NDArray]) -> float:
        """Helper to calculate expectation value with arbitrary MPO."""
        if not self.mps.mps:
            raise ValueError("MPS not initialized")

        expectation = np.array([[[1.0]]])
        for i, mpo_tensor in enumerate(mpo_tensors):
            expectation = self.contract_left(
                mpo_tensor, self.mps.mps[i], expectation, self.mps.mps[i]
            )
        return float(expectation[0][0][0])


def main() -> None:
    """Main function to demonstrate DMRG calculation."""
    # Model parameters
    LOCAL_DIM = 2  # Local bond dimension
    NUM_SITES = 10  # Number of sites
    BOND_DIM = 10  # Bond dimension for DMRG
    NUM_SWEEPS = 8  # Number of DMRG sweeps

    # Initialize quantum state components
    mpo = MPO(local_dim=LOCAL_DIM, num_sites=NUM_SITES)
    mps = MPS(local_dim=LOCAL_DIM, num_sites=NUM_SITES)

    # Create and run Heisenberg model
    model = HeisenbergModel(mps=mps, mpo=mpo, bond_dim=BOND_DIM, num_sweeps=NUM_SWEEPS)

    try:
        # Run DMRG optimization
        model.run_dmrg()

        # Calculate and display results
        final_energy = model.calculate_expectation()
        print("\nFinal Results:")
        print(f"Energy expectation value: {final_energy:16.12f}")

        variance = model.calculate_variance()
        print(f"Energy variance: {variance:16.12f}")

    except Exception as e:
        print(f"Error during DMRG calculation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
