"""Differentiable age-structured predator-prey environment for exp0526."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from .config import EcologyConfig, EnvConfig, SpeciesProfileConfig
except ImportError:  # pragma: no cover
    from config import EcologyConfig, EnvConfig, SpeciesProfileConfig


@dataclass
class DerivedSpecies:
    zeta: torch.Tensor
    n: torch.Tensor
    kappa: torch.Tensor
    pi0: torch.Tensor


@dataclass
class DerivedEnvironment:
    species: tuple[DerivedSpecies, DerivedSpecies]
    gamma1: torch.Tensor
    gamma2: torch.Tensor
    x1_star0: torch.Tensor
    x2_star0: torch.Tensor
    profiles: dict[str, torch.Tensor]
    scalars: dict[str, float]


class AgeStructuredPredatorPreyEnv:
    def __init__(self, env_config: EnvConfig, ecology_config: EcologyConfig, *, device: torch.device) -> None:
        self.config = env_config
        self.ecology = ecology_config
        self.device = device
        self.age = torch.linspace(0.0, env_config.age_max, env_config.num_age, device=device)
        self.da = env_config.age_max / float(env_config.num_age - 1)
        self.weights = torch.full((env_config.num_age,), self.da, device=device)
        self.weights[0] *= 0.5
        self.weights[-1] *= 0.5
        self.cfl = env_config.dt / self.da

    def _mixed_signal(self, step: int, *, phase: float) -> float:
        numerator = 0.0
        denom = 0.0
        t = float(step)
        for freq, amp in self.ecology.mixed_sin_components:
            numerator += amp * torch.sin(torch.tensor(freq * t + phase)).item()
            denom += abs(amp)
        return numerator / max(denom, 1e-12)

    def _modulated(self, base: float, strength: float, signal: float) -> float:
        return max(base * (1.0 + strength * signal), self.config.positivity_eps)

    def _species_profiles(self, profile: SpeciesProfileConfig, *, step: int, species_idx: int) -> dict[str, torch.Tensor | float]:
        sig_k = self._mixed_signal(step, phase=0.7 * species_idx + 0.0)
        sig_mb = self._mixed_signal(step, phase=0.7 * species_idx + 1.9)
        sig_ms = self._mixed_signal(step, phase=0.7 * species_idx + 3.1)
        k_amp = self._modulated(profile.k_amp, self.ecology.k_amp_strength, sig_k)
        mu_base = self._modulated(profile.mu_base, self.ecology.mu_base_strength, sig_mb)
        mu_sen_amp = self._modulated(profile.mu_sen_amp, self.ecology.mu_sen_amp_strength, sig_ms)

        age = self.age
        k = profile.k_base + k_amp * torch.exp(-((age - profile.k_center) ** 2) / (2.0 * profile.k_sigma ** 2))
        mu = mu_base + profile.mu_juv_amp * torch.exp(-profile.mu_juv * age) + mu_sen_amp * age ** profile.mu_sen
        g = profile.g_base + profile.g_amp * torch.exp(-((age - profile.g_center) ** 2) / (2.0 * profile.g_sigma ** 2))
        return {
            "k": k,
            "mu": mu,
            "g": g,
            "k_amp": float(k_amp),
            "mu_base": float(mu_base),
            "mu_sen_amp": float(mu_sen_amp),
        }

    def profiles(self, step: int) -> dict[str, torch.Tensor]:
        rows = [self._species_profiles(profile, step=step, species_idx=idx) for idx, profile in enumerate(self.config.species)]
        return {
            "k": torch.stack([rows[0]["k"], rows[1]["k"]]),  # type: ignore[list-item]
            "mu": torch.stack([rows[0]["mu"], rows[1]["mu"]]),  # type: ignore[list-item]
            "g": torch.stack([rows[0]["g"], rows[1]["g"]]),  # type: ignore[list-item]
        }

    def _integral(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * self.weights)

    def _cumulative_integral(self, values: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(values * self.da, dim=0)
        return cumulative - 0.5 * values * self.da

    def _lotka_residual(self, zeta: torch.Tensor, k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        exponent = -self._cumulative_integral(mu + zeta)
        return self._integral(k * torch.exp(exponent)) - 1.0

    def solve_zeta(self, k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._lotka_residual(torch.tensor(0.0, device=self.device), k, mu).item() <= 0.0:
                return torch.tensor(0.0, device=self.device)
            lo = torch.tensor(0.0, device=self.device)
            hi = torch.tensor(1.0, device=self.device)
            while self._lotka_residual(hi, k, mu).item() > 0.0 and hi.item() < 128.0:
                hi = hi * 2.0
            for _ in range(self.config.root_bisect_iters):
                mid = 0.5 * (lo + hi)
                if self._lotka_residual(mid, k, mu).item() > 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

    def _derive_species(self, k: torch.Tensor, mu: torch.Tensor) -> DerivedSpecies:
        zeta = self.solve_zeta(k, mu)
        cumulative = self._cumulative_integral(mu + zeta)
        n = torch.exp(-cumulative).detach()
        reverse = torch.flip(torch.cumsum(torch.flip(k * n * self.weights, dims=(0,)), dim=0), dims=(0,))
        pi0 = (reverse / torch.clamp(n, min=self.config.positivity_eps)).detach()
        # Use the discrete identity <pi0, n> = kappa so that x = x* gives eta = 0
        # under the same quadrature used during rollouts.
        kappa = self._integral(pi0 * n).detach()
        return DerivedSpecies(zeta=zeta.detach(), n=n, kappa=kappa, pi0=pi0)

    def derive(self, step: int) -> DerivedEnvironment:
        with torch.no_grad():
            rows = [self._species_profiles(profile, step=step, species_idx=idx) for idx, profile in enumerate(self.config.species)]
            k = torch.stack([rows[0]["k"], rows[1]["k"]])  # type: ignore[list-item]
            mu = torch.stack([rows[0]["mu"], rows[1]["mu"]])  # type: ignore[list-item]
            g = torch.stack([rows[0]["g"], rows[1]["g"]])  # type: ignore[list-item]
            sp1 = self._derive_species(k[0], mu[0])
            sp2 = self._derive_species(k[1], mu[1])
            gamma1 = self._integral(g[0] * sp2.n).detach()
            gamma2 = self._integral(g[1] * sp1.n).detach()
            x1_star0 = self.config.equilibrium_x1_multiplier / torch.clamp(sp2.zeta * gamma2, min=self.config.positivity_eps)
            x2_star0 = (sp1.zeta - sp2.zeta + 1.0 / torch.clamp(x1_star0 * gamma2, min=self.config.positivity_eps)) / torch.clamp(gamma1, min=self.config.positivity_eps)
            x2_star0 = torch.clamp(x2_star0, min=self.config.positivity_eps)
            scalars = {
                "zeta1": float(sp1.zeta.item()),
                "zeta2": float(sp2.zeta.item()),
                "x1_star0": float(x1_star0.item()),
                "x2_star0": float(x2_star0.item()),
                "k1_amp": float(rows[0]["k_amp"]),
                "k2_amp": float(rows[1]["k_amp"]),
                "mu1_base": float(rows[0]["mu_base"]),
                "mu2_base": float(rows[1]["mu_base"]),
                "mu1_sen_amp": float(rows[0]["mu_sen_amp"]),
                "mu2_sen_amp": float(rows[1]["mu_sen_amp"]),
            }
            return DerivedEnvironment(
                species=(sp1, sp2),
                gamma1=gamma1,
                gamma2=gamma2,
                x1_star0=x1_star0.detach(),
                x2_star0=x2_star0.detach(),
                profiles={"k": k.detach(), "mu": mu.detach(), "g": g.detach()},
                scalars=scalars,
            )

    def equilibrium_state(self, step: int) -> torch.Tensor:
        derived = self.derive(step)
        x1 = derived.x1_star0 * derived.species[0].n
        x2 = derived.x2_star0 * derived.species[1].n
        return torch.stack([x1, x2]).detach()

    def initial_state(self) -> torch.Tensor:
        equilibrium = self.equilibrium_state(0)
        scales = torch.tensor([self.config.species[0].init_scale, self.config.species[1].init_scale], device=self.device).view(2, 1)
        return (equilibrium * scales).detach()

    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, DerivedEnvironment]:
        derived = self.derive(step)
        numerator1 = torch.sum(derived.species[0].pi0 * state[0] * self.weights)
        numerator2 = torch.sum(derived.species[1].pi0 * state[1] * self.weights)
        denom1 = torch.clamp(derived.x1_star0 * derived.species[0].kappa, min=self.config.positivity_eps)
        denom2 = torch.clamp(derived.x2_star0 * derived.species[1].kappa, min=self.config.positivity_eps)
        eta = torch.stack([
            torch.log(torch.clamp(numerator1, min=self.config.positivity_eps) / denom1),
            torch.log(torch.clamp(numerator2, min=self.config.positivity_eps) / denom2),
        ])
        return eta.view(1, 2), derived

    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, DerivedEnvironment]:
        derived = self.derive(step)
        profiles = derived.profiles
        k = profiles["k"]
        mu = profiles["mu"]
        g = profiles["g"]
        u_scalar = u.reshape(())
        interaction1 = torch.sum(g[0] * state[1] * self.weights)
        interaction2_den = torch.clamp(torch.sum(g[1] * state[0] * self.weights), min=self.config.positivity_eps)
        reaction = torch.stack([
            mu[0] + u_scalar + interaction1,
            mu[1] + u_scalar + 1.0 / interaction2_den,
        ])

        transported = (1.0 - self.cfl) * state[:, 1:] + self.cfl * state[:, :-1]
        interior = transported * torch.exp(-self.config.dt * reaction[:, 1:])
        boundary = torch.sum(k * state * self.weights.view(1, -1), dim=1, keepdim=True)
        next_state = torch.cat([boundary, interior], dim=1)
        next_state = torch.clamp(next_state, min=self.config.positivity_eps)
        return next_state, derived

    def diagnostics(self, state: torch.Tensor, derived: DerivedEnvironment) -> dict[str, float]:
        population1 = self._integral(state[0])
        population2 = self._integral(state[1])
        equilibrium_population1 = self._integral(derived.x1_star0 * derived.species[0].n)
        equilibrium_population2 = self._integral(derived.x2_star0 * derived.species[1].n)
        return {
            "min_population": float(torch.min(state).detach().cpu().item()),
            "population1": float(population1.detach().cpu().item()),
            "population2": float(population2.detach().cpu().item()),
            "equilibrium_population1": float(equilibrium_population1.detach().cpu().item()),
            "equilibrium_population2": float(equilibrium_population2.detach().cpu().item()),
            **derived.scalars,
        }
