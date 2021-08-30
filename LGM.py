from copy import deepcopy
import torch
from typing import Tuple
import numpy as np

from piecewise_functions import PiecewiseFunction, SegmentFunction

torch.manual_seed(0)


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

    def _capital_lambda(self, end: float) -> PiecewiseFunction:
        lambda_function = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[- torch.ones(1) * l] for l in self.lambda_structure]
        )
        exp_lambda_forward = lambda_function.get_exponential_forward_integral(start=0)
        exp_lambda_forward_p = exp_lambda_forward.antiderivative()
        exp_lambda_backward = (-1 * lambda_function).get_exponential_forward_integral(start=0)
        epsilon = 1e-9  # hack used to compute the integral, as exp_lambda_forward_p.__call__(x) chooses the segment
            # function by idx such that idx is the last index where term_structure[idx] >= x
        capital_lambda = exp_lambda_backward * (exp_lambda_forward_p(end - epsilon) - exp_lambda_forward_p)

        return capital_lambda

    def _r(self, t: float, forward_measure: float = None) -> torch.Tensor:
        constant_term_integral = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * l] for l in self.lambda_structure]
        )
        constant_term = self.r_0 * torch.exp(- constant_term_integral.integral(0, t))

        drift_internal_integral = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[- torch.ones(1) * l for l in self.lambda_structure]
        )
        drift_exp_integral = drift_internal_integral.get_exponential_backward_integral(end=t)
        drift_integral_constants = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * l * m] for l, m in zip(self.lambda_structure, self.m_structure)]
        )
        drift_function = drift_exp_integral * drift_integral_constants
        drift_term = drift_function.integral(0, t)

        vol_exp_integral = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[- torch.ones(1) * l] for l in self.lambda_structure]
        ).get_exponential_backward_integral(end=t)
        vol_term_function = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * s] for s in self.sigma_structure]
        )
        vol_function = vol_exp_integral * vol_term_function
        vol_term = (vol_function ** 2).integral(0, t)
        r_std = torch.sqrt(vol_term)

        if forward_measure is not None:
            # we do a change of measure to the t-forward measure (where t is the "forward_measure" variable)
            sigma_d = -1 * self._capital_lambda(forward_measure) * vol_term_function
            measure_change_drift_function = vol_function * sigma_d
            drift_term += measure_change_drift_function.integral(0, t)

        r = constant_term + drift_term + r_std * self.dW
        return r

    def _r_integral_params(
            self,
            t: float,
            T: float,
            forward_measure: float = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        r_t = self._r(t, forward_measure=forward_measure)
        capital_lambda = self._capital_lambda(T)
        constant_term = capital_lambda(t) * r_t

        drift_term_capital_lambda = self._capital_lambda(T)
        lambda_x_m = PiecewiseFunction(

            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * l * m] for l, m in zip(self.lambda_structure, self.m_structure)]
        )
        drift_term = (drift_term_capital_lambda * lambda_x_m).integral(t, T)

        vol_term_capital_lambda = self._capital_lambda(T)
        vol_term_function = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * s] for s in self.sigma_structure]
        )

        vol_function = vol_term_capital_lambda * vol_term_function
        vol_term = (vol_function ** 2).integral(t, T)
        r_integral_std = torch.sqrt(vol_term)

        if forward_measure is not None:
            sigma_d = -1 * self._capital_lambda(forward_measure) * vol_term_function
            measure_change_drift_function = sigma_d * vol_function
            drift_term += measure_change_drift_function.integral(t, T)

        return constant_term, drift_term, r_integral_std

    def r_integral(self, t: float, T: float, forward_measure: float = None):
        constant_term, drift_term, r_integral_std = self._r_integral_params(t, T, forward_measure=forward_measure)
        return constant_term + drift_term + r_integral_std * self.dW

    def discount_factor(self, t: float, T: float, forward_measure: float = None):
        constant_term, drift_term, r_integral_std = self._r_integral_params(t, T, forward_measure=forward_measure)
        return torch.exp(- constant_term - drift_term + 0.5 * r_integral_std ** 2)

    def libor(self, T, T_tau, coverage=None):
        if coverage is None:
            coverage = (T_tau - T)
        libors = (1 - self.discount_factor(T, T_tau, forward_measure=T_tau)) / coverage
        return libors

    @staticmethod
    def call_payoff(L: torch.Tensor, K: float) -> torch.Tensor:
        return torch.maximum(L - K, torch.zeros(L.shape))

    @staticmethod
    def put_payoff(L: torch.Tensor, K: float) -> torch.Tensor:
        return torch.maximum(K - L, torch.zeros(L.shape))

    def price_caplet(self, T: float, T_tau: float, K: float) -> torch.Tensor:
        L = self.libor(T, T_tau)
        pv = torch.mean(self.call_payoff(L, K))
        return pv

    def price_floorlet(self, T: float, T_tau: float, K: float) -> torch.Tensor:
        L = self.libor(T, T_tau)
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

    res = lgm.get_caplet_greeks(1, 1.25, 0.01)
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
        'n_paths': 1000000
    }

    # test_lgm(case_1, "Flat curve, no vol")
    # flat curve, add vol
    case_2 = deepcopy(case_1)
    case_2['sigma_structure'] = np.ones(4) * 1e-2
    test_lgm(case_2, f"Flat curve, vol = {case_2['sigma_structure'][0]}")

    # # step function m, no vol
    # case_3 = deepcopy(case_1)
    # case_3['m_structure'] = np.arange(4)
    # test_lgm(case_3, "Step function m, no vol")
    #
    # # step function m, with vol
    # case_4 = deepcopy(case_3)
    # case_4['sigma_structure'] = np.ones(4) * 1e-3
    # test_lgm(case_4, f"Step function m, vol = {case_4['sigma_structure'][0]}")


if __name__ == '__main__':
    run_tests()
