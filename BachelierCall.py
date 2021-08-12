import numpy as np
import torch

torch.manual_seed(12345)


class BachelierModel:
    def __init__(
            self,
            s_0 : np.array,
            sigma_term_structure: np.array,
            term_structure: np.array,
            n_paths: int = 10000
    ):
        self.s_0 = torch.from_numpy(s_0)
        self.s_0.requires_grad_()
        full_sigma = np.hstack([sigma_term_structure, np.array([sigma_term_structure[-1]])])
        self.sigma = torch.from_numpy(full_sigma)
        self.sigma.requires_grad_()
        self.n_paths = n_paths
        self.term_structure = self.clean_term_structure(term_structure)
        self.dW = torch.randn(size=(n_paths,))

        self._payoff_mapping = {
            'call': self.call_payoff
        }

    @staticmethod
    def clean_term_structure(term_structure: np.array) -> np.array:
        if term_structure[0] != 0:
            term_structure = np.hstack([np.zeros(1), term_structure])
        if term_structure[-1] != np.inf:
            term_structure = np.hstack([term_structure, np.array([np.inf])])
        return term_structure

    def final_underlying_price(self, t: float) -> torch.Tensor:
        term_structure_mask = self.term_structure <= t
        term_structure_jump_times = self.term_structure[term_structure_mask]
        time_remainder = t - term_structure_jump_times[-1]
        time_periods = np.diff(term_structure_jump_times)
        time_periods = torch.from_numpy(time_periods)
        period_variances = (self.sigma[term_structure_mask[:-1]][:-1] ** 2) * time_periods
        total_variance = torch.sum(period_variances) + (self.sigma[term_structure_mask[:-1]][-1] ** 2) * time_remainder
        s_f_vol = torch.sqrt(total_variance)
        s_f = self.s_0 + s_f_vol * self.dW
        return s_f

    @staticmethod
    def call_payoff(s_f: torch.Tensor, k: float) -> torch.Tensor:
        payoff = torch.maximum(s_f - k, torch.zeros(s_f.shape))
        return payoff

    def price(self, k: float, t: float, option_type: str = 'call') -> torch.Tensor:
        s_f = self.final_underlying_price(t)
        payoff_function = self._payoff_mapping.get(option_type, None)
        assert payoff_function is not None
        payoff = payoff_function(s_f, k)
        return torch.mean(payoff)
    

def test():
    n_paths = 100000
    s_0 = np.array([1], dtype=np.float64)
    sigma = np.linspace(0.1, 4, 30, dtype=np.float64)
    term_structure = np.exp(np.linspace(0, 3, 30))
    option_type = 'call'
    k = 1
    t = 5
    model = BachelierModel(s_0, sigma, term_structure, n_paths)
    pv = model.price(k, t, option_type)
    pv.backward()
    delta = model.s_0.grad
    vega = model.sigma.grad
    total_vega = torch.sum(vega).item()

    print(f'delta: {delta} \n\n  vega: {vega} \n\n total_vega: {total_vega}')
    return


if __name__ == '__main__':
    test()


