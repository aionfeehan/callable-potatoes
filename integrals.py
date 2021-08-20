import numpy as np
import torch
from typing import Callable, List, Union, Tuple

torch.manual_seed(0)


class TorchPolynomial:
    def __init__(self, coefficients: torch.Tensor):
        self.coefficients = coefficients.double()
        self.degree = len(coefficients) - 1

    def __str__(self):
        return str(self.coefficients)

    def __getitem__(self, item: int):
        assert item <= self.degree
        return self.coefficients[item]

    def __add__(self, other):
        polynome = self.coerce_to_polynomial(other)
        padded_1, padded_2 = self.pad_zeros(self.coefficients, polynome.coefficients)
        return TorchPolynomial(padded_1 + padded_2)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        polynome = self.coerce_to_polynomial(other)
        padded_1, padded_2 = self.pad_zeros(self.coefficients, polynome.coefficients)
        return TorchPolynomial(padded_1 - padded_2)

    def __rsub__(self, other):
        return self.coerce_to_polynomial(other).__sub__(self)

    def __mul__(self, other):
        polynome = self.coerce_to_polynomial(other)
        new_coefs = []
        new_deg = self.degree + polynome.degree
        for k in range(new_deg + 1):
            new_coefs.append(torch.sum([self[i] * polynome[k - i] for i in range(k)]))
        return TorchPolynomial(torch.stack(new_coefs, dim=0))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, t: float) -> torch.Tensor:
        t_powers = torch.tensor([t ** k for k in range(self.degree + 1)], dtype=torch.float64)
        return torch.dot(self.coefficients, t_powers)

    def __eq__(self, other):
        other_p = self.coerce_to_polynomial(other)
        if other_p.degree != self.degree:
            return False
        else:
            return all(other_p[k].float() == self[k].float() for k in range(self.degree + 1))

    def derivative(self):
        new_coefs = torch.stack([k * self[k] for k in range(1, self.degree + 1)], dim=0)
        return TorchPolynomial(new_coefs)

    def antiderivative(self, constant: float = 0):
        new_coefficients = [torch.tensor(constant, dtype=torch.float64)]
        new_coefficients.extend([coef / (k + 1) for k, coef in enumerate(self.coefficients)])
        return TorchPolynomial(torch.stack(new_coefficients, dim=0))

    @staticmethod
    def coerce_to_polynomial(other):
        if isinstance(other, TorchPolynomial):
            return other
        elif isinstance(other, float) or isinstance(other, int):
            return TorchPolynomial(torch.Tensor([other], dtype=torch.float64))
        else:
            raise TypeError(f"Unsupported type {type(other)}")

    @staticmethod
    def pad_zeros(polynome_1, polynome_2):
        deg_1, deg_2 = len(polynome_1), len(polynome_2)
        max_deg = max((deg_1, deg_2))
        res_1 = torch.nn.functional(polynome_1, (0, max_deg - deg_1), "constant", 0)
        res_2 = torch.nn.functional(polynome_2, (0, max_deg - deg_2), "constant", 0)
        return res_1, res_2


class SegmentFunction:
    # Represent functions like sum_i^n[P_i(x) * exp(alpha_i * x)]
    def __init__(
            self,
            exp_coefs: torch.Tensor,  # tensor of alpha_i coefs, size n
            polynomes: Union[List[TorchPolynomial], List[torch.Tensor]]  # list of tensors of polynomial coefs
    ):
        self.exp_coefs = self.coerce_to_tensor(exp_coefs)
        if isinstance(polynomes[0], torch.Tensor):
            self.polynomes = [TorchPolynomial(p) for p in polynomes]
        elif isinstance(polynomes[0], TorchPolynomial):
            self.polynomes = polynomes

    def __add__(self, other):
        other_function = self.coerce_to_segment_function(other)
        return SegmentFunction(
            torch.cat([self.exp_coefs, other.exp_coefs]),
            self.polynomes + other.polynomes
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_function = self.coerce_to_segment_function(other)
        return SegmentFunction(
            torch.cat([self.exp_coefs, other.exp_coefs]),
            self.polynomes + [- k for k in other.polynomes]
        )

    def __mul__(self, other):
        other_function = self.coerce_to_segment_function(other)
        new_polynomes = []
        exp_coefs = []
        for i, poly_1, in enumerate(self.polynomes):
            for j, poly_2 in enumerate(other.polynomes):
                new_polynomes.append(poly_1 * poly_2)
                exp_coefs.append(self.exp_coefs[i] + other.exp_coefs[j])
        return SegmentFunction(
            torch.stack(exp_coefs, dim=0),
            new_polynomes
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, t):
        return torch.sum(torch.stack([p(t) * torch.exp(self.exp_coefs[k] * t) for k, p in enumerate(self.exp_coefs)]))

    def derivative(self):
        exp_coefs = torch.stack([self.exp_coefs.clone(), self.exp_coefs.clone()])
        polynomes = [p.derivative() for p in self.polynomes] + \
                    [coef * self.polynomes[k] for k, coef in enumerate(self.exp_coefs)]
        return SegmentFunction(exp_coefs, polynomes)

    def antiderivative(self):
        new_polynomes = [self._single_antiderivative(p, self.exp_coefs[k]) for k, p in enumerate(self.polynomes)]
        return SegmentFunction(
            self.exp_coefs.clone(),
            new_polynomes
        )

    def _single_antiderivative(self, polynome, exp_coef):
        while polynome.degree > 0:
            return polynome * 1 / exp_coef + self._single_antiderivative(polynome.derivative() * 1 / exp_coef, exp_coef)
        return polynome * 1 / exp_coef

    @staticmethod
    def coerce_to_segment_function(other):
        if isinstance(other, SegmentFunction):
            return other
        elif isinstance(other, float) or isinstance(other, int):
            return SegmentFunction(torch.tensor([0], dtype=torch.float64), torch.tensor([other], dtype=torch.float64))
        else:
            raise TypeError(f"Unsupported type  {type(other)}")

    @staticmethod
    def coerce_to_tensor(exp_coefs):
        if isinstance(exp_coefs, torch.Tensor):
            return exp_coefs.reshape(-1)
        elif isinstance(exp_coefs, list) or isinstance(exp_coefs, tuple):
            return torch.tensor(exp_coefs, dtype=torch.float64)
        elif isinstance(exp_coefs, float) or isinstance(exp_coefs, int):
            return torch.tensor([exp_coefs], dtype=torch.float64)
        else:
            raise TypeError(f"Unsupported type {type(exp_coefs)}")


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


def test_polynomes():
    case_1 = {
        "coefficients": torch.tensor([1, 1, 1])
    }
    p = TorchPolynomial(**case_1)
    assert p(0) == 1
    assert p(1) == 3
    assert p.derivative() == TorchPolynomial(torch.tensor([1, 2]))
    assert p.antiderivative(1) == TorchPolynomial(torch.tensor([1, 1, 1/2, 1/3]))

    print("Polynomial tests passed")
    return


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
    test_polynomes()
    test_constant()
    test_exponential()
