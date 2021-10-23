import torch
from typing import Tuple
import numpy as np

from piecewise_functions import PiecewiseFunction

torch.manual_seed(0)


class LGM1F:
    def __init__(
            self,
            term_structure: np.ndarray,
            lambda_structure: np.ndarray,
            sigma_structure: np.ndarray,
            spot_discount_factor_function: callable = None,
            m_structure: np.ndarray = None,
            r_0: np.ndarray = None,
            n_paths: int = 10000
    ):
        self.term_structure = torch.from_numpy(term_structure.astype(np.float64))
        self.lambda_structure = torch.from_numpy(lambda_structure.astype(np.float64))
        self.lambda_structure.requires_grad_()
        self.sigma_structure = torch.from_numpy(sigma_structure.astype(np.float64))
        self.sigma_structure.requires_grad_()
        self.m_structure = self._build_m_structure(spot_discount_factor_function, m_structure)
        self.r_0 = self.m_structure[0].clone()
        self.r_0.requires_grad_()
        self.m_structure.requires_grad_()
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
            polynomial_structure=[[- torch.ones(1) * lambda_] for lambda_ in self.lambda_structure]
        )
        exp_lambda_forward = lambda_function.get_exponential_forward_integral(start=0)
        exp_lambda_forward_p = exp_lambda_forward.antiderivative()
        exp_lambda_backward = (-1 * lambda_function).get_exponential_forward_integral(start=0)
        epsilon = 1e-15
        # hack used to compute the integral, as exp_lambda_forward_p.__call__(x) chooses the segment
        # function by idx such that idx is the last index where term_structure[idx] >= x
        capital_lambda = exp_lambda_backward * (exp_lambda_forward_p(end - epsilon) - exp_lambda_forward_p)

        return capital_lambda

    def _d_capital_lambda(self, end: float) -> PiecewiseFunction:
        lambda_function = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[-torch.ones(1) * lambda_] for lambda_ in self.lambda_structure]
        )
        exp_lambda_backward = (-1 * lambda_function).get_exponential_forward_integral(start=end)
        return exp_lambda_backward

    def _r(self, t: float, forward_measure: float = None) -> torch.Tensor:
        constant_term_function = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[torch.ones(1) * lambda_] for lambda_ in self.lambda_structure]
        )
        constant_term = self.r_0 * torch.exp(- constant_term_function.integral(0, t))

        drift_internal_integral = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[- torch.ones(1) * lambda_ for lambda_ in self.lambda_structure]
        )
        drift_exp_integral = drift_internal_integral.get_exponential_backward_integral(end=t)
        drift_integral_constants = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[
                [torch.ones(1) * lambda_ * m] for lambda_, m in zip(self.lambda_structure, self.m_structure)
            ]
        )
        drift_function = drift_exp_integral * drift_integral_constants
        drift_term = drift_function.integral(0, t)

        vol_exp_integral = PiecewiseFunction(
            term_structure=self.term_structure,
            exp_coef_structure=None,
            polynomial_structure=[[- torch.ones(1) * lambda_] for lambda_ in self.lambda_structure]
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
            polynomial_structure=[
                [torch.ones(1) * lambda_ * m] for lambda_, m in zip(self.lambda_structure, self.m_structure)
            ]
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

    def _build_m_structure(self, discount_factor_function: callable, m_structure: np.ndarray) -> torch.Tensor:
        assert (discount_factor_function is not None) or (m_structure is not None)
        if m_structure is not None:
            assert len(m_structure) + 1 == len(self.term_structure)
            return torch.from_numpy(m_structure.astype(np.float64))
        else:
            term_structure = self.term_structure.clone()
            term_structure.requires_grad_()
            discount_factors = torch.stack([discount_factor_function(t) for t in term_structure])
            ln_df = torch.log(discount_factors)
            d_ln_df = torch.autograd.grad(
                outputs=[d for d in ln_df],
                inputs=term_structure,
                only_inputs=True,
                allow_unused=True,
                create_graph=True
            )[0]
            dd_ln_df = torch.autograd.grad(
                outputs=[d for d in d_ln_df],
                inputs=term_structure,
                only_inputs=True,
                allow_unused=True,
                retain_graph=True
            )[0]
            ms = []
            for k, t in enumerate(self.term_structure):
                lambda_ = self.lambda_structure[k]
                d_capital_lambda = self._d_capital_lambda(t)
                sigma_function = PiecewiseFunction(
                    term_structure=term_structure,
                    exp_coef_structure=None,
                    polynomial_structure=[[- torch.ones(1) * s] for s in self.sigma_structure]
                )
                remainder = ((d_capital_lambda * sigma_function) ** 2).integral(0, t)
                ms.append(- dd_ln_df[k] / lambda_ - d_ln_df[k] + remainder / lambda_)
            return torch.stack(ms)

    def discount_factor(self, t: float, T: float, forward_measure: float = None):
        constant_term, drift_term, r_integral_std = self._r_integral_params(t, T, forward_measure=forward_measure)
        return torch.exp(- constant_term - drift_term + 0.5 * r_integral_std ** 2)

    def libor(self, T, T_tau, coverage=None):
        if coverage is None:
            coverage = (T_tau - T)
        libors = (1 / self.discount_factor(T, T_tau, forward_measure=T) - 1) / coverage
        return libors

    def price_caplet(self, T:float, T_tau: float, K: float, coverage: float = None) -> torch.Tensor:
        if coverage is None:
            coverage = T_tau - T
        df = torch.mean(self.discount_factor(0, T))
        df_forward = self.discount_factor(T, T_tau, forward_measure=T)
        libor_rate = self.libor(T, T_tau)
        payoff = torch.maximum(libor_rate - K, torch.zeros(libor_rate.shape))
        pv = df * coverage * torch.mean(df_forward * payoff)
        return pv

    def price_floorlet(self, T: float, T_tau: float, K: float, coverage: float = None) -> torch.Tensor:
        if coverage is None:
            coverage = T_tau - T
        libor_rate = self.libor(T, T_tau)
        df = torch.mean(self.discount_factor(0, T))
        df_forward = self.discount_factor(T, T_tau, forward_measure=T)
        payoff = torch.maximum(K - libor_rate, torch.zeros(libor_rate.shape))
        pv = df * coverage * torch.mean(df_forward * payoff)
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
    df_04 = torch.mean(lgm.discount_factor(0, 4)).item()
    df_24 = torch.mean(lgm.discount_factor(2, 4)).item()
    df_34 = torch.mean(lgm.discount_factor(3, 4)).item()
    df_03 = torch.mean(lgm.discount_factor(0, 3)).item()

    print(f"Case {test_name} ")
    print(f"DF 0-1: {df_01}")
    print(f"DF 3-4: {df_34}")
    print(f"DF 0-2: {df_02}")
    print(f"DF 2-4: {df_24}")

    T = 1
    T_tau = 1.25
    forward = (torch.mean(lgm.discount_factor(0, T)) / torch.mean(lgm.discount_factor(0, T_tau)) - 1) / (T_tau - T)
    print(f"Forward: {forward} \n")

    res = lgm.get_caplet_greeks(T, T_tau, forward)
    print(res)
    print('\n\n')

    return


def test_m_calibration():
    m_structure = torch.ones(5) * 1e-2
    discount_function = PiecewiseFunction(
        term_structure=torch.arange(5),
        exp_coef_structure=None,
        polynomial_structure=[[-torch.ones(1) * m] for m in m_structure]
    ).get_exponential_forward_integral(start=0)

    case_1 = {
        'r_0': np.array([0.01]),
        'term_structure': np.arange(5),
        'lambda_structure': np.ones(5) * 1e-2,
        'spot_discount_factor_function': discount_function,
        'sigma_structure': np.ones(5) * 0,
        'n_paths': 10000
    }

    lgm = LGM1F(**case_1)

    df_01 = torch.mean(lgm.discount_factor(0, 1))
    df_02 = torch.mean(lgm.discount_factor(0, 2))

    print(f'df01: {discount_function(1)}, -----------> df01_fit: {df_01}')
    print(f'df02: {discount_function(2)}, -----------> df02_fit: {df_02}')

    return


def run_tests():

    # test_lgm(case_1, "Flat curve, no vol")
    # flat curve, add vol
    # case_2 = deepcopy(case_1)
    # case_2['sigma_structure'] = np.ones(4) * 1e-2
    # test_lgm(case_2, f"Flat curve, vol = {case_2['sigma_structure'][0]}")

    # # step function m, no vol
    # case_3 = deepcopy(case_1)
    # case_3['m_structure'] = np.arange(4)
    # test_lgm(case_3, "Step function m, no vol")
    #
    # # step function m, with vol
    # case_4 = deepcopy(case_3)
    # case_4['sigma_structure'] = np.ones(4) * 1e-3
    # test_lgm(case_4, f"Step function m, vol = {case_4['sigma_structure'][0]}")
    test_m_calibration()


if __name__ == '__main__':
    run_tests()
