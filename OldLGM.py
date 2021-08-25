from copy import deepcopy
import torch
from typing import Tuple
import numpy as np

from piecewise_functions import OldPiecewiseIntegral


class LGM1F:
    def __init__(
            self,
            r_0: np.ndarray,
            term_structure: np.ndarray,
            lambda_structure: np.ndarray,
            m_structure: np.ndarray,
            sigma_structure: np.ndarray,
            n_paths: int = 10000
    ):
        self.r_0 = torch.from_numpy(r_0.astype(np.float64))
        self.r_0.requires_grad_()
        self.term_structure = torch.from_numpy(term_structure.astype(np.float64))
        self.lambda_structure = torch.from_numpy(lambda_structure.astype(np.float64))
        self.lambda_structure.requires_grad_()
        self.m_structure = torch.from_numpy(m_structure.astype(np.float64))
        self.m_structure.requires_grad_()
        self.sigma_structure = torch.from_numpy(sigma_structure.astype(np.float64))
        self.sigma_structure.requires_grad_()
        self.n_paths = n_paths
        self.dW = torch.randn(n_paths, dtype=torch.float64)

    def _truncate_term_structure(self, a: float, b: float, structure: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # get term structure defined over [a, b] and corresponding mask for coefficients
        is_in_segment = (self.term_structure >= a) & (self.term_structure < b)
        used_term_structure = self.term_structure[is_in_segment]
        if structure is not None:
            used_coefs = structure[is_in_segment[:-1]]
        else:
            used_coefs = None

        if a not in used_term_structure:
            used_term_structure = torch.nn.functional.pad(used_term_structure, (1, 0), "constant", a)
            if a < self.term_structure[0] and used_coefs is not None:
                    used_coefs = torch.nn.functional.pad(used_coefs, (1, 0), "constant", used_coefs[0])

        if b not in used_term_structure:
            used_term_structure = torch.nn.functional.pad(used_term_structure, (0, 1), "constant", b)
            if b > self.term_structure[-1] and used_coefs is not None:
                used_coefs = torch.nn.functional.pad(used_coefs, (1, 0), "constant", used_coefs[-1])

        return used_term_structure, used_coefs

    def _capital_lambda(self, t: float, T: float) -> torch.Tensor:
        if t == T:
            return torch.tensor(0, dtype=torch.float64)
        elif t > T:
            return - self._capital_lambda(T, t)
        else:
            lambda_integral = OldPiecewiseIntegral(
                term_structure=self.term_structure,
                coef_structure=self.lambda_structure
            )
            used_term_structure, coef_structure = self._truncate_term_structure(t, T, - self.lambda_structure)
            constant_structure = torch.exp(
                - torch.stack([lambda_integral(t, k) for k in used_term_structure[:-1]], dim=0)
            ) * torch.exp(-coef_structure * used_term_structure[:-1])

            capital_lambda = OldPiecewiseIntegral(
                term_structure=used_term_structure,
                coef_structure=coef_structure,
                constant_structure=constant_structure,
                _type='exponential'
            )
        return capital_lambda(t, T)

    def _r(self, t: float) -> torch.Tensor:
        constant_term_integral = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=self.lambda_structure,
            _type='constant'
        )
        constant_term = self.r_0 * torch.exp(- constant_term_integral(t))

        drift_term_constants_1 = self.lambda_structure * self.m_structure
        drift_term_exp_integral = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=self.lambda_structure,
            _type="constant"
        )
        drift_term_constants_2 = torch.exp(
            - torch.stack(
                [drift_term_exp_integral(s, t) for s in self.term_structure[1:]],
                dim=0
            )
        )
        drift_term_constants_3 = torch.exp(self.lambda_structure * self.term_structure[:-1])
        drift_term_constants = drift_term_constants_1 * drift_term_constants_2 * drift_term_constants_3
        drift_term_integral = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=-self.lambda_structure,
            constant_structure=drift_term_constants,
            _type="exponential"
        )
        drift_term = drift_term_integral(0, t)

        vol_term_exp_integral = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=self.lambda_structure,
            _type="constant"
        )
        vol_term_constants_2 = torch.exp(
            - torch.stack(
                [vol_term_exp_integral(s, t) for s in self.term_structure[1:]],
                dim=0
            )
        )
        vol_term_constants = (self.sigma_structure * vol_term_constants_2) ** 2
        vol_term_integral = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=- 2 * self.lambda_structure,
            constant_structure=vol_term_constants,
            _type="exponential"
        )
        vol_term = vol_term_integral(0, t)
        r_std = torch.sqrt(vol_term)

        r = constant_term + drift_term + r_std * self.dW
        return r

    def _r_integral_params(self, t, T):
        r_t = self._r(t)
        constant_term = self._capital_lambda(t, T) * r_t
        lambda_x_m = self.lambda_structure * self.m_structure
        drift_term_integral_1 = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=lambda_x_m/self.lambda_structure,
            _type="constant"
        )
        capital_lambdas = torch.stack(
            [self._capital_lambda(t_i, T) for t_i in self.term_structure[1:]],
            dim=0
        )
        exp_lambda_t_i = torch.exp(- self.lambda_structure * self.term_structure[1:])
        drift_term_integral_2 = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=self.lambda_structure,
            constant_structure=(1 / self.lambda_structure - capital_lambdas) * exp_lambda_t_i * lambda_x_m,
            _type="exponential"
        )

        drift_term = drift_term_integral_1(t, T) - drift_term_integral_2(t, T)

        vol_term_integral_1 = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=(self.sigma_structure / self.lambda_structure) ** 2,
            _type="constant"
        )

        vol_2_constants = 2 * (capital_lambdas / self.lambda_structure - 1 / self.lambda_structure) * \
                          exp_lambda_t_i * (self.sigma_structure ** 2)

        vol_term_integral_2 = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=self.lambda_structure,
            constant_structure=vol_2_constants,
            _type="exponential"
        )

        vol_3_constants = (1 / self.lambda_structure ** 2 + \
                          capital_lambdas ** 2 - \
                          2 * capital_lambdas / self.lambda_structure) * \
                          (exp_lambda_t_i ** 2) * \
                          self.sigma_structure ** 2

        vol_term_integral_3 = OldPiecewiseIntegral(
            term_structure=self.term_structure,
            coef_structure=2 * self.lambda_structure,
            constant_structure=vol_3_constants,
            _type="exponential"
        )

        vol_term = vol_term_integral_1(t, T) + vol_term_integral_2(t, T) + vol_term_integral_3(t, T)
        r_integral_std = torch.sqrt(vol_term)

        return constant_term, drift_term, r_integral_std

    def r_integral(self, t, T):
        constant_term, drift_term, r_integral_std = self._r_integral_params(t, T)
        return constant_term + drift_term + r_integral_std * self.dW

    def discount_factor(self, t, T):
        constant_term, drift_term, r_integral_std = self._r_integral_params(t, T)
        return torch.exp(- constant_term - drift_term + 0.5 * r_integral_std ** 2)

    def libor(self, t, T, T_tau, coverage=None):
        if coverage is None:
            coverage = (T_tau - T)
        libors = (1 / self.discount_factor(T, T_tau) - 1) / coverage
        return libors

    @staticmethod
    def call_payoff(L: torch.Tensor, K: float) -> torch.Tensor:
        return torch.maximum(L - K, torch.zeros(L.shape))

    @staticmethod
    def put_payoff(L: torch.Tensor, K: float) -> torch.Tensor:
        return torch.maximum(K - L, torch.zeros(L.shape))

    def price_caplet(self, T: float, T_tau: float, K: float) -> torch.Tensor:
        L = self.libor(0, T, T_tau)
        pv = torch.mean(self.call_payoff(L, K))
        return pv

    def price_floorlet(self, T: float, T_tau: float, K: float) -> torch.Tensor:
        L = self.libor(0, T, T_tau)
        pv = torch.mean(self.put_payoff(L, K))
        return pv

    def reset_grad(self):
        self.r_0.grad = None
        self.sigma_structure.grad = None
        self.lambda_structure.grad = None
        self.m_structure.grad = None
        return

    def get_caplet_greeks(self, T: float, T_tau: float, K: float) -> dict:
        self.reset_grad()
        pv = self.price_caplet(T, T_tau, K)
        pv.backward()
        res = {
            'pv': pv.item(),
            'dr_0': self.r_0.grad,
            'dsigma': self.sigma_structure.grad,
            'dlambda': self.lambda_structure.grad,
            'dm': self.m_structure.grad
        }
        return res


def test_lgm(test_case, test_name="test"):
    lgm = LGM1F(**test_case)
    df_01 = torch.mean(lgm.discount_factor(0, 1)).item()
    df_02 = torch.mean(lgm.discount_factor(0, 2)).item()
    df_24 = torch.mean(lgm.discount_factor(2, 4)).item()
    df_34 = torch.mean(lgm.discount_factor(3, 4)).item()

    print(f"Case {test_name} ")
    print(f"DF 0-1: {df_01}")
    print(f"DF 3-4: {df_34}")
    print(f"DF 0-2: {df_02}")
    print(f"DF 2-4: {df_24}")
    print('\n\n')

    res = lgm.get_caplet_greeks(1, 2, 0.01)
    print(res)
    print('\n\n')

    return


def run_tests():
    # flat curve, no vol
    case_1 = {
        'r_0': np.array([0.01]),
        'term_structure': np.arange(5),
        'lambda_structure': np.ones(4) * 1,
        'm_structure': np.ones(4) * 0.01,
        'sigma_structure': np.ones(4) * 1e-9,
        'n_paths': 10000
    }

    test_lgm(case_1, "Flat curve, no vol")
    # flat curve, add vol
    case_2 = deepcopy(case_1)
    case_2['sigma_structure'] = np.ones(4) * 1e-3
    test_lgm(case_2, f"Flat curve, vol = {case_2['sigma_structure'][0]}")

    # step function m, no vol
    case_3 = deepcopy(case_1)
    case_3['m_structure'] = np.arange(4)
    test_lgm(case_3, "Step function m, no vol")

    # step function m, with vol
    case_4 = deepcopy(case_3)
    case_4['sigma_structure'] = np.ones(4) * 1e-3
    test_lgm(case_4, f"Step function m, vol = {case_4['sigma_structure'][0]}")


if __name__ == '__main__':
    run_tests()
