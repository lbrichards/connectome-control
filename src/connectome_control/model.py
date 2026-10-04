"""A continuous-time recurrent network whose wiring is the C. elegans connectome.

The connectome supplies topology, not weights: the adjacency pattern becomes a
fixed binary mask over a trainable weight matrix. Synapse counts from the data
set the initial weight magnitudes; signs are learned.

Dynamics (graded-potential / non-spiking, matching real C. elegans neurons):

    tau * dh/dt = -h + W_chem @ tanh(h) + gap(h) + W_in @ x + b
    gap(h)_i    = sum_j G_ij * (h_j - h_i)

Chemical synapses are directed and signed. Gap junctions are symmetric, have
non-negative conductance, and act on voltage difference rather than on the
presynaptic output -- that distinction is what makes this connectome-faithful
rather than merely sparse.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


class ConnectomeRNN(nn.Module):
    def __init__(
        self,
        chem: np.ndarray,
        gap: np.ndarray,
        in_idx: np.ndarray,
        out_idx: np.ndarray,
        n_in: int = 1,
        n_out: int = 1,
        dt_over_tau_init: float = 0.5,
        in_gain: float = 2.0,
        spectral_radius: float = 1.2,
        ablate_recurrence: bool = False,
        memoryless: bool = False,
        head_hidden: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n = chem.shape[0]
        self.n = n
        # `memoryless` is the stricter control: it also discards the previous
        # state each step. Zeroing the recurrent synapses alone leaves every
        # neuron as an independent leaky integrator, and a bank of low-pass
        # filters with learnable time constants can approximate a derivative --
        # so that ablation does NOT remove all memory.
        self.memoryless = memoryless
        self.ablate = ablate_recurrence or memoryless

        m_chem = torch.from_numpy((chem > 0).astype(np.float32))
        m_gap = torch.from_numpy((gap > 0).astype(np.float32))
        if ablate_recurrence:
            m_chem = torch.zeros_like(m_chem)
            m_gap = torch.zeros_like(m_gap)
        self.register_buffer("m_chem", m_chem)
        self.register_buffer("m_gap", m_gap)

        # Initial magnitudes from synapse counts (log-scaled: counts span 1..142),
        # random signs, then rescaled so the recurrent Jacobian is well conditioned.
        mag = np.log1p(chem).astype(np.float32)
        sign = np.where(np.random.default_rng(seed).random(chem.shape) < 0.5, -1.0, 1.0)
        w0 = (mag * sign).astype(np.float32)
        self._w0 = w0            # rescaled below, once leak and dt/tau are known

        # Gap conductances are non-negative; parameterised through softplus and
        # symmetrised on every forward pass. Row sums are normalised to O(1):
        # they enter the leak term, and leaving them raw makes the system stiff.
        gmag = np.log1p(gap).astype(np.float32)
        scale = max(float(gmag.sum(1).mean()), 1e-6)
        gmag = gmag / scale
        self.w_gap_raw = nn.Parameter(
            torch.from_numpy(np.log(np.expm1(np.clip(gmag, 1e-4, None))).astype(np.float32))
        )

        self.register_buffer("in_idx", torch.from_numpy(in_idx))
        self.register_buffer("out_idx", torch.from_numpy(out_idx))

        self.w_in = nn.Parameter(torch.randn(len(in_idx), n_in, generator=g) * in_gain)
        self.bias = nn.Parameter(torch.zeros(n))

        # Per-neuron time constant, kept in (0, 1) as dt/tau via sigmoid.
        p = float(np.clip(dt_over_tau_init, 1e-3, 1 - 1e-3))
        self.dt_tau_raw = nn.Parameter(torch.full((n,), float(np.log(p / (1 - p)))))

        # A purely linear readout cannot express the pump<->catch mode switch
        # of a hybrid controller; head_hidden > 0 inserts one small hidden
        # layer. The recurrent core is untouched -- this is head capacity only.
        if head_hidden > 0:
            self.readout = nn.Sequential(
                nn.Linear(len(out_idx), head_hidden), nn.Tanh(),
                nn.Linear(head_hidden, n_out))
        else:
            self.readout = nn.Linear(len(out_idx), n_out)

        # Scale the chemical weights so the *effective* recurrent Jacobian --
        # not W on its own -- sits at the requested spectral radius. Gap
        # conductance enters the leak, so ignoring it leaves the network
        # strongly contracting and unable to retain state.
        self._calibrate(spectral_radius)

    def _effective_rho(self, scale: float) -> float:
        with torch.no_grad():
            w = (self._w0 * scale) * self.m_chem.numpy()
            g = self.gap_matrix().numpy()
            leak = 1.0 + g.sum(1)
            decay = np.exp(-torch.sigmoid(self.dt_tau_raw).numpy() * leak)
            j = np.diag(decay) + (((1.0 - decay) / leak)[:, None] * (w + g))
            return _spectral_radius(j)

    def _calibrate(self, target: float, iters: int = 24) -> None:
        if self.ablate:
            self.w_chem = nn.Parameter(torch.from_numpy(self._w0 * 0.0))
            return
        lo, hi = 0.0, 8.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self._effective_rho(mid) < target:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        self.w_chem = nn.Parameter(torch.from_numpy((self._w0 * scale).astype(np.float32)))
        self.init_rho = self._effective_rho(scale)

    def gap_matrix(self) -> torch.Tensor:
        g = torch.nn.functional.softplus(self.w_gap_raw) * self.m_gap
        return 0.5 * (g + g.T)

    def precompute(self) -> tuple[torch.Tensor, ...]:
        """Terms that are constant across a rollout.

        The self-terms (membrane leak plus total gap conductance) are collected
        into `leak` and integrated exactly; the rest is treated as input over
        the step. Explicit Euler is unstable here because gap conductance can
        make dt/tau * leak exceed 2.
        """
        w_chem = self.w_chem * self.m_chem
        g = self.gap_matrix()
        leak = 1.0 + g.sum(1)
        decay = torch.exp(-torch.sigmoid(self.dt_tau_raw) * leak)
        return w_chem, g, leak, decay

    def input_drive(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, n_in) -> (B, N) external current, injected at sensory nodes."""
        d = torch.zeros(x_t.shape[0], self.n, device=x_t.device, dtype=x_t.dtype)
        d[:, self.in_idx] = x_t @ self.w_in.T
        return d

    def cell(self, h: torch.Tensor, drive_t: torch.Tensor, pre) -> torch.Tensor:
        w_chem, g, leak, decay = pre
        if self.memoryless:
            return (drive_t + self.bias) / leak
        i_in = torch.tanh(h) @ w_chem.T + h @ g.T + drive_t + self.bias
        return h * decay + (i_in / leak) * (1.0 - decay)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        return self.readout(torch.tanh(h[:, self.out_idx]))

    def init_state(self, b: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(b, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_in) -> (B, T, n_out)"""
        b, t, _ = x.shape
        h = self.init_state(b, x)
        pre = self.precompute()
        outs = []
        for k in range(t):
            h = self.cell(h, self.input_drive(x[:, k]), pre)
            outs.append(self.read(h))
        return torch.stack(outs, dim=1)

    def n_trainable_synapses(self) -> int:
        return int(self.m_chem.sum().item() + self.m_gap.sum().item() // 2)
