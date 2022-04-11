"""
See "Fine-Tune your smile - correction to Hagan et al." by Jan Obloj, Imperial College London, 2007

Slight change of notations - our "alpha" is their "nu", our "sigma_0" is their "alpha"

We implement to include a shift parameter "zeta".
"""


import numpy as np
import torch


class TorchSabr:
    def __init__(
            self,
            atm_vol,
            sigma_0,
            alpha,
            beta,
            rho,
            zeta
    ):
        self.atm_vol = torch.tensor(atm_vol, requires_grad=True)
        self.sigma_0 = torch.tensor(sigma_0, requires_grad=True)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.beta = torch.tensor(beta, requires_grad=True)
        self.rho = torch.tensor(rho, requires_grad=True)
        self.zeta = torch.tensor(zeta, requires_grad=True)

    def __call__(
            self,
            forward,
            strike,
            maturity
    ):
        s = forward - self.zeta
        x = torch.log(s / strike)

        if x > 0:
            z = self.alpha / self.sigma_0 * (s ** (1 - self.beta) - strike ** (1 - self.beta)) / (1 - self.beta)
            I_0 = self.alpha * x / torch.log((torch.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

        else:
            I_0 = self.sigma_0 * strike ** (self.beta - 1)

        I_1H = (
                ((self.beta - 1) ** 2) *
                (self.alpha ** 2) +
                6 * self.rho * self.sigma_0 * self.alpha * self.beta +
                2 * self.alpha ** 2 -
                3 * (self.rho * self.alpha) ** 2
        )

        implied_vol = I_0 * (1 + I_1H * maturity)


        return implied_vol


if __name__ == '__main__':

    atm_vol = 1.0168 / 100
    atm_vol = 47.406 / 100
    sigma_0 = 1.0022
    alpha = 0.485
    beta = 0.5121
    rho = 0.0825
    zeta = 0.0

    test_sabr = TorchSabr(
        atm_vol=atm_vol,
        sigma_0=sigma_0,
        alpha=alpha,
        beta=beta,
        rho=rho,
        zeta=zeta
    )
    forward = 0.021
    strike = 0.021
    maturity = 1
    print(test_sabr(forward, strike, maturity))

