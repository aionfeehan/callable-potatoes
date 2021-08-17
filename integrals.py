import numpy as np
import torch
from typing import Callable, Tuple

torch.manual_seed(0)


class PiecewiseIntegral:
    # Compute functions as piecewise constant integrals
    def __init__(
            self,
            term_structure: torch.Tensor,
            coef_structure: torch.Tensor,
            constant_structure: torch.Tensor = None,
            _type: str = 'constant'
    ):
        assert len(term_structure) == len(coef_structure) + 1
        assert constant_structure is None or (len(constant_structure) == len(coef_structure))
        self.term_structure = term_structure.double()
        self.coef_structure = coef_structure.double()
        self.constant_structure = constant_structure.double() if constant_structure is not None else constant_structure
        self._type = _type
        if _type == 'constant':
            self.integral_function = self._integrate_constant()
        elif _type == 'exponential':
            self.integral_function = self._integrate_exponential()

    def __call__(self, a, b=None):
        if b is None:
            b = a
            a = 0
        if a > b:
            return - self.integral_function(b, a)
        else:
            return self.integral_function(a, b)

    def _integrate_constant(self) -> Callable:

        def integrate_function(a: float, b: float) -> float:
            used_term_structure, used_coefs, _ = self._truncate_term_structure(a, b)
            segment_integrals = used_coefs * torch.diff(used_term_structure)
            res = torch.sum(segment_integrals)
            return res

        return integrate_function

    def _integrate_exponential(self) -> Callable:

        def integrate_function(a: float, b: float) -> float:
            used_term_structure, used_coefs, used_constant_structure = self._truncate_term_structure(a, b)
            if used_constant_structure is None:
                used_constant_structure = 1
            segment_end_values = torch.exp(used_coefs * used_term_structure[1:])
            segment_start_values = torch.exp(used_coefs * used_term_structure[:-1])
            used_coefs_denominator = used_coefs.clone()
            used_coefs_denominator[used_coefs == 0] = 1
            constant_factor = used_constant_structure / used_coefs_denominator
            segment_integrals = (segment_end_values - segment_start_values) * constant_factor
            res = torch.sum(segment_integrals)
            return res

        return integrate_function

    def _truncate_term_structure(self, a: float, b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get term structure defined over [a, b] and corresponding mask for coefficients
        is_in_segment = (self.term_structure >= a) & (self.term_structure < b)
        used_term_structure = self.term_structure[is_in_segment]
        used_coefs = self.coef_structure[is_in_segment[:-1]]
        if self.constant_structure is not None:
            used_constants = self.constant_structure[is_in_segment[:-1]]
        else:
            used_constants = None

        if a not in used_term_structure:
            used_term_structure = torch.nn.functional.pad(used_term_structure, (1, 0), "constant", a)
            if a < self.term_structure[0]:
                used_coefs = torch.nn.functional.pad(used_coefs, (1,  0), "constant", used_coefs[0])
                if used_constants is not None:
                    used_constants = torch.nn.functional.pad(used_constants, (1, 0), "constant", used_constants[0])

        if b not in used_term_structure:
            used_term_structure = torch.nn.functional.pad(used_term_structure, (0, 1), "constant", b)
            if b > self.term_structure[-1]:
                used_coefs = torch.nn.functional.pad(used_coefs, (1, 0), "constant", used_coefs[-1])
                if used_constants is not None:
                    used_constants = torch.nn.functional.pad(used_constants, (0, 1), "constant", used_constants[-1])

        return used_term_structure, used_coefs, used_constants


def test_constant():
    case_1 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.ones(4, dtype=torch.float64)
    }
    case_2 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.arange(4, dtype=torch.float64)
    }

    integral_1 = PiecewiseIntegral(**case_1)
    assert integral_1(0, 1).item() == 1
    assert integral_1(0, 2.5).item() == 2.5
    assert integral_1(-1, 2).item() == 3
    assert integral_1(0, 6).item() == 6

    integral_2 = PiecewiseIntegral(**case_2)
    assert integral_2(0, 1).item() == 0
    assert integral_2(0, 2.5).item() == 2
    assert integral_2(-1, 2).item() == 1
    assert integral_2(3, 6).item() == 9

    print('All constant test cases passed')
    return


def test_exponential():
    case_1 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.ones(4, dtype=torch.float64),
        "constant_structure": torch.ones(4, dtype=torch.float64),
        "_type": 'exponential'
    }
    case_2 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.arange(4, dtype=torch.float64),
        "constant_structure": torch.arange(4, dtype=torch.float64),
        "_type": "exponential"
    }
    integral_1 = PiecewiseIntegral(**case_1)
    assert integral_1(0, 1).item() == np.exp(1) - 1
    assert integral_1(0, 4.5).item() == np.exp(4.5) - 1

    integral_2 = PiecewiseIntegral(**case_2)
    assert integral_2(0, 1).item() == 0
    assert integral_2(0, 2).item() == np.exp(2) - np.exp(1)
    assert integral_2(0, 3).item() == (np.exp(2) - np.exp(1)) + 2 * (np.exp(3 * 2) - np.exp(2 * 2)) / 2

    print("All exponential test cases passed")
    return


if __name__ == "__main__":
    test_constant()
    test_exponential()
