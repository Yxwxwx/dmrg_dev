import numpy as np
import scipy.sparse.linalg


class MPO:
    """Matrix Product Operator (MPO) for representing Hamiltonians."""

    def __init__(self, local_dim, num_sites):
        self.local_dim = local_dim
        self.num_sites = num_sites
        self.mpo = self.construct_mpo()

    def construct_mpo(self):
        """Constructs MPO for the Heisenberg Hamiltonian."""
        # Define local operators
        I = np.identity(2)
        Z = np.zeros((2, 2))
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 0], [1, 0]])
        Sm = np.array([[0, 1], [0, 0]])

        # MPO tensors for each site in the chain
        W = np.array(
            [
                [I, Z, Z, Z, Z],
                [Sp, Z, Z, Z, Z],
                [Sm, Z, Z, Z, Z],
                [Sz, Z, Z, Z, Z],
                [Z, 0.5 * Sm, 0.5 * Sp, Sz, I],
            ]
        )
        Wfirst = np.array([[Z, 0.5 * Sm, 0.5 * Sp, Sz, I]])
        Wlast = np.array([[I], [Sp], [Sm], [Sz], [Z]])

        return [Wfirst] + ([W] * (self.num_sites - 2)) + [Wlast]

    def product(self, other):
        """Multiplies two MPOs, element-wise."""
        assert len(self.mpo) == len(other.mpo)
        product_mpo = []
        for i in range(len(self.mpo)):
            product_mpo.append(self._product_w(self.mpo[i], other.mpo[i]))
        return product_mpo

    @staticmethod
    def _product_w(W1, W2):
        """Helper function to combine two MPO tensors."""
        return np.reshape(
            np.einsum("abst,cdtu->acbdsu", W1, W2, optimize=True),
            [
                W1.shape[0] * W2.shape[0],
                W1.shape[1] * W2.shape[1],
                W1.shape[2],
                W2.shape[3],
            ],
        )

    def coarse_grain(self, W, X):
        """Coarse-grains two site MPO tensors into a single tensor."""
        return np.reshape(
            np.einsum("abst,bcuv->acsutv", W, X, optimize=True),
            [W.shape[0], X.shape[1], W.shape[2] * X.shape[2], W.shape[3] * X.shape[3]],
        )


class MPS:
    """Matrix Product State (MPS) representing the state of the system."""

    def __init__(self, local_dim, num_sites):
        self.local_dim = local_dim
        self.num_sites = num_sites
        self.mps = self.initialize_state()

    def initialize_state(self):
        """Initializes the MPS state to |01010101...>"""
        InitialA1 = np.zeros((self.local_dim, 1, 1))
        InitialA1[0, 0, 0] = 1
        InitialA2 = np.zeros((self.local_dim, 1, 1))
        InitialA2[1, 0, 0] = 1
        return [InitialA1, InitialA2] * int(self.num_sites / 2)

    def initial_e(self, W):
        """Initializes the left boundary E matrix for the vacuum state."""
        E = np.zeros((W.shape[0], 1, 1))
        E[0] = 1
        return E

    def initial_f(self, W):
        """Initializes the right boundary F matrix for the vacuum state."""
        F = np.zeros((W.shape[1], 1, 1))
        F[-1] = 1
        return F

    def coarse_grain(self, A, B):
        """Coarse-grains two MPS sites into one."""
        return np.reshape(
            np.einsum("sij,tjk->stik", A, B, optimize=True),
            [A.shape[0] * B.shape[0], A.shape[1], B.shape[2]],
        )

    def fine_grain(self, A, dims):
        """Performs fine-graining on MPS, splitting a single coarse-grained site."""
        assert A.shape[0] == dims[0] * dims[1]
        Theta = np.transpose(
            np.reshape(A, dims + [A.shape[1], A.shape[2]]), (0, 2, 1, 3)
        )
        M = np.reshape(Theta, (dims[0] * A.shape[1], dims[1] * A.shape[2]))
        U, S, V = np.linalg.svd(M, full_matrices=False)
        U = np.reshape(U, (dims[0], A.shape[1], -1))
        V = np.transpose(np.reshape(V, (-1, dims[1], A.shape[2])), (1, 0, 2))
        return U, S, V

    def truncate(self, U, S, V, m):
        """Truncates MPS tensors by retaining the m largest singular values."""
        m = min(len(S), m)
        trunc = np.sum(S[m:])
        S = S[:m]
        U = U[:, :, :m]
        V = V[:, :m, :]
        return U, S, V, trunc, m


class HeisenbergModel:
    """Heisenberg model that implements the two-site DMRG algorithm."""

    def __init__(self, mps, mpo, bond_dim, num_sweeps):
        self.mps = mps
        self.mpo = mpo
        self.bond_dim = bond_dim
        self.num_sweeps = num_sweeps

    def contract_from_left(self, W, A, E, B):
        """Contracts tensors from the left side."""
        Temp = np.einsum("sij,aik->sajk", A, E, optimize=True)
        Temp = np.einsum("sajk,abst->tbjk", Temp, W, optimize=True)
        return np.einsum("tbjk,tkl->bjl", Temp, B, optimize=True)

    def contract_from_right(self, W, A, F, B):
        """Contracts tensors from the right side."""
        Temp = np.einsum("sij,bjl->sbil", A, F, optimize=True)
        Temp = np.einsum("sbil,abst->tail", Temp, W, optimize=True)
        return np.einsum("tail,tkl->aik", Temp, B, optimize=True)

    def construct_f(self):
        """Constructs F-matrices used for contractions from the right."""
        F = [self.mps.initial_f(self.mpo.mpo[-1])]
        for i in range(len(self.mpo.mpo) - 1, 0, -1):
            F.append(
                self.contract_from_right(
                    self.mpo.mpo[i], self.mps.mps[i], F[-1], self.mps.mps[i]
                )
            )
        return F

    def construct_e(self):
        """Constructs E-matrices used for contractions from the left."""
        return [self.mps.initial_e(self.mpo.mpo[0])]

    def optimize_two_sites(self, A, B, W1, W2, E, F, m, direction):
        """Optimizes two-site tensors to minimize energy."""
        W = self.mpo.coarse_grain(W1, W2)
        AA = self.mps.coarse_grain(A, B)
        H = HamiltonianMultiply(E, W, F)
        E, V = scipy.sparse.linalg.eigsh(H, 1, v0=AA, which="SA")
        AA = np.reshape(V[:, 0], H.req_shape)
        A, S, B = self.mps.fine_grain(AA, [A.shape[0], B.shape[0]])
        A, S, B, trunc, m = self.mps.truncate(A, S, B, m)
        if direction == "right":
            B = np.einsum("ij,sjk->sik", np.diag(S), B, optimize=True)
        else:
            assert direction == "left"
            A = np.einsum("sij,jk->sik", A, np.diag(S), optimize=True)
        return E[0], A, B, trunc, m

    def run_dmrg(self):
        """Runs the two-site DMRG algorithm."""
        E = self.construct_e()
        F = self.construct_f()
        F.pop()
        for sweep in range(int(self.num_sweeps / 2)):
            for i in range(0, len(self.mps.mps) - 2):
                Energy, self.mps.mps[i], self.mps.mps[i + 1], trunc, states = (
                    self.optimize_two_sites(
                        self.mps.mps[i],
                        self.mps.mps[i + 1],
                        self.mpo.mpo[i],
                        self.mpo.mpo[i + 1],
                        E[-1],
                        F[-1],
                        self.bond_dim,
                        "right",
                    )
                )
                print(
                    f"Sweep {sweep*2} Sites {i},{i+1}    Energy {Energy:16.12f}    "
                    f"States {states:4} Truncation {trunc:16.12f}"
                )
                # Update the E matrices for the left contraction
                E.append(
                    self.contract_from_left(
                        self.mpo.mpo[i], self.mps.mps[i], E[-1], self.mps.mps[i]
                    )
                )
                F.pop()

            for i in range(len(self.mps.mps) - 2, 0, -1):
                Energy, self.mps.mps[i], self.mps.mps[i + 1], trunc, states = (
                    self.optimize_two_sites(
                        self.mps.mps[i],
                        self.mps.mps[i + 1],
                        self.mpo.mpo[i],
                        self.mpo.mpo[i + 1],
                        E[-1],
                        F[-1],
                        self.bond_dim,
                        "left",
                    )
                )
                print(
                    f"Sweep {sweep*2 + 1} Sites {i},{i+1}    Energy {Energy:16.12f}    "
                    f"States {states:4} Truncation {trunc:16.12f}"
                )
                # Update the F matrices for the right contraction
                F.append(
                    self.contract_from_right(
                        self.mpo.mpo[i + 1],
                        self.mps.mps[i + 1],
                        F[-1],
                        self.mps.mps[i + 1],
                    )
                )
                E.pop()
        return self.mps.mps

    def expectation_value(self):
        """Calculates the expectation value of the Hamiltonian for the MPS."""
        E = [[[1]]]
        for i in range(len(self.mpo.mpo)):
            E = self.contract_from_left(
                self.mpo.mpo[i], self.mps.mps[i], E, self.mps.mps[i]
            )
        return E[0][0][0]

    def calculate_variance(self):
        """Calculates the variance to measure the accuracy of the ground state energy."""
        HamSquared = self.mpo.product(self.mpo)  # Square of the Hamiltonian operator
        energy = self.expectation_value()
        H2 = self.expectation_value_mps(HamSquared)
        return H2 - energy**2

    def expectation_value_mps(self, mpo):
        """Helper to calculate expectation value with arbitrary MPO."""
        E = [[[1]]]
        for i in range(len(mpo)):
            E = self.contract_from_left(mpo[i], self.mps.mps[i], E, self.mps.mps[i])
        return E[0][0][0]


class HamiltonianMultiply(scipy.sparse.linalg.LinearOperator):
    """Class for Hamiltonian-vector multiplication for eigensolver."""

    def __init__(self, E, W, F):
        self.E = E
        self.W = W
        self.F = F
        self.req_shape = [W.shape[2], E.shape[1], F.shape[1]]
        self.size = self.req_shape[0] * self.req_shape[1] * self.req_shape[2]
        super().__init__(dtype=np.dtype("d"), shape=(self.size, self.size))

    def _matvec(self, A):
        """Matrix-vector product with Hamiltonian MPO representation."""
        R = np.einsum(
            "aij,sik->ajsk", self.E, np.reshape(A, self.req_shape), optimize=True
        )
        R = np.einsum("ajsk,abst->bjtk", R, self.W, optimize=True)
        R = np.einsum("bjtk,bkl->tjl", R, self.F, optimize=True)
        return np.reshape(R, -1)


if __name__ == "__main__":
    # Parameters
    d = 2  # Local bond dimension
    N = 10  # Number of sites
    D = 10  # Bond dimension for DMRG

    # Initialize MPO, MPS, and Heisenberg Model
    mpo = MPO(local_dim=d, num_sites=N)
    mps = MPS(local_dim=d, num_sites=N)
    heisenberg_model = HeisenbergModel(mps=mps, mpo=mpo, bond_dim=D, num_sweeps=8)

    # Run DMRG algorithm
    optimized_mps = heisenberg_model.run_dmrg()

    # Calculate final energy expectation and variance
    final_energy = heisenberg_model.expectation_value()
    print(f"Final energy expectation value: {final_energy}")

    variance = heisenberg_model.calculate_variance()
    print(f"Variance: {variance:16.12f}")
