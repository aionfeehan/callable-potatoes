import numpy as np
import torch
from typing import Callable, List, Union, Tuple

torch.manual_seed(0)


class TorchPolynomial:
    def __init__(self, coefficients: torch.Tensor):
        if isinstance(coefficients, TorchPolynomial):
            self.coefficients = coefficients.coefficients
            self.degree = coefficients.degree
        else:
            self.coefficients = coefficients.double().reshape(-1)
            self.degree = len(self.coefficients) - 1

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
            valid_coefs = torch.cat(
                [torch.stack([self[i] * polynome[j] for j in range(min(i, polynome.degree) + 1)])
                 for i in range(min(k, self.degree) + 1)]
            )
            new_coefs.append(torch.sum(valid_coefs))
        return TorchPolynomial(torch.stack(new_coefs, dim=0))

    def __pow__(self, power):
        assert isinstance(power, int)
        if power == 1:
            return self
        elif power == 0:
            return TorchPolynomial(torch.ones(1))
        else:
            return self.__pow__(power // 2) * self.__pow__(power - power // 2)

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
            return TorchPolynomial(torch.tensor([other], dtype=torch.float64))
        elif isinstance(other, torch.Tensor):
            assert (not other.size()) or (len(other) == 1)
            return TorchPolynomial(other)
        else:
            raise TypeError(f"Unsupported type {type(other)}")

    @staticmethod
    def pad_zeros(polynome_1, polynome_2):
        deg_1, deg_2 = len(polynome_1), len(polynome_2)
        max_deg = max((deg_1, deg_2))
        res_1 = torch.nn.functional.pad(polynome_1, (0, max_deg - deg_1), "constant", 0)
        res_2 = torch.nn.functional.pad(polynome_2, (0, max_deg - deg_2), "constant", 0)
        return res_1, res_2


class SegmentFunction:
    # Represent functions like sum_i^n[P_i(x) * exp(alpha_i * x)]
    def __init__(
            self,
            exp_coefs: torch.Tensor,  # tensor of alpha_i coefs, size n
            polynomes: Union[List[TorchPolynomial], List[torch.Tensor]]  # list of tensors of polynomial coefs
    ):
        if exp_coefs is None:
            exp_coefs = torch.zeros(len(polynomes))
        self.exp_coefs = self.coerce_to_tensor(exp_coefs)
        if isinstance(polynomes[0], torch.Tensor):
            self.polynomes = [TorchPolynomial(p) for p in polynomes]
        elif isinstance(polynomes[0], TorchPolynomial):
            self.polynomes = polynomes
        self._align_by_exp_coef()

    def __str__(self):
        return "\n".join([f"{(str(p), self.exp_coefs[k])}" for k, p in enumerate(self.polynomes)])

    def __add__(self, other):
        other_function = self.coerce_to_segment_function(other)
        return SegmentFunction(
            torch.cat([self.exp_coefs, other_function.exp_coefs]),
            self.polynomes + other_function.polynomes
        )

    def __pow__(self, power):
        assert isinstance(power, int)
        if power == 1:
            return self
        elif power == 0:
            return SegmentFunction(
                exp_coefs=torch.zeros(1),
                polynomes=[TorchPolynomial(torch.ones(1))]
            )
        else:
            return self.__pow__(power // 2) * self.__pow__(power - power // 2)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_function = self.coerce_to_segment_function(other)
        return SegmentFunction(
            torch.cat([self.exp_coefs, other_function.exp_coefs]),
            self.polynomes + [-1 * k for k in other_function.polynomes]
        )

    def __rsub__(self, other):
        return - 1 * self.__sub__(other)

    def __mul__(self, other):
        other_function = self.coerce_to_segment_function(other)
        new_polynomes = []
        exp_coefs = []
        for i, poly_1, in enumerate(self.polynomes):
            for j, poly_2 in enumerate(other_function.polynomes):
                new_polynomes.append(poly_1 * poly_2)
                exp_coefs.append(self.exp_coefs[i] + other_function.exp_coefs[j])
        return SegmentFunction(
            torch.stack(exp_coefs, dim=0),
            new_polynomes
        )

    @property
    def degree(self):
        return max([p.degree for p in self.polynomes])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, t):
        return torch.sum(torch.stack([p(t) * torch.exp(self.exp_coefs[k] * t) for k, p in enumerate(self.polynomes)]))

    def __eq__(self, other):
        other_sf = self.coerce_to_segment_function(other)
        if len(other_sf.exp_coefs) != len(self.exp_coefs):
            return False
        else:
            order = np.argsort(self.exp_coefs.detach())
            polynomes_eq = all(self.polynomes[k] == other.polynomes[k] for k in order)
            coefs_eq = all(self.exp_coefs[k] == other.exp_coefs[k] for k in order)
            return polynomes_eq and coefs_eq

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
        if exp_coef == 0:
            return polynome.antiderivative(0)
        else:
            while polynome.degree > 0:
                return polynome * 1 / exp_coef - \
                       self._single_antiderivative(polynome.derivative() * 1 / exp_coef, exp_coef)
            return polynome * 1 / exp_coef

    def _align_by_exp_coef(self):
        if len(self.exp_coefs) == len(np.unique(self.exp_coefs.detach())):
            return
        else:
            values, counts = np.unique(self.exp_coefs.detach(), return_counts=True)
            to_drop = []
            for k, v in enumerate(values):
                if counts[k] > 1:
                    is_value = (np.array(self.exp_coefs.detach()) == v).nonzero()[0]
                    idx_to_keep = is_value[0]
                    for i in range(1, counts[k]):
                        idx_to_drop = is_value[i]
                        self.polynomes[idx_to_keep] += self.polynomes[idx_to_drop]
                        to_drop.append(idx_to_drop)
            to_keep = [k for k in range(len(self.exp_coefs)) if k not in to_drop]
            self.exp_coefs = self.exp_coefs[to_keep]
            self.polynomes = [self.polynomes[k] for k in to_keep]
        return

    @staticmethod
    def coerce_to_segment_function(other):
        if isinstance(other, SegmentFunction):
            return other
        elif isinstance(other, float) or isinstance(other, int):
            return SegmentFunction(torch.tensor([0], dtype=torch.float64), [torch.tensor([other], dtype=torch.float64)])
        elif isinstance(other, torch.Tensor):
            assert len(other.reshape(-1)) == 1, "Must be single value if passing tensor"
            return SegmentFunction(torch.tensor([0], dtype=torch.float64), [other])
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

    def get_exponential(self):
        assert self.degree <= 1
        assert all(self.exp_coefs == 0)
        constants = []
        exps = torch.zeros(len(self.polynomes), dtype=torch.float64)
        for k, p in enumerate(self.polynomes):
            constants.append(TorchPolynomial(torch.exp(p[0])))
            if p.degree == 1:
                exps[k] += p[1]
        exp_segment_function = SegmentFunction(
            exp_coefs=exps,
            polynomes=constants
        )
        return exp_segment_function


class PiecewiseFunction:
    def __init__(
            self,
            term_structure: torch.Tensor,
            exp_coef_structure: List[torch.Tensor] = None,
            polynomial_structure: Union[List[List[torch.Tensor]], List[List[TorchPolynomial]]] = None,
            segment_functions: List[SegmentFunction] = None
    ):
        assert (polynomial_structure is None) or (len(polynomial_structure) == len(term_structure))
        assert (exp_coef_structure is None) or (len(exp_coef_structure)) == len(term_structure)

        self.term_structure = term_structure.double()
        if segment_functions is None:
            if exp_coef_structure is None:
                exp_coef_structure = [torch.zeros(len(p_list), dtype=torch.float64) for p_list in polynomial_structure]
            self.exp_coef_structure = exp_coef_structure
            self.polynomial_structure = [[TorchPolynomial(p) for p in seg_polys] for seg_polys in polynomial_structure]
            if polynomial_structure is None:
                polynomial_structure = [[torch.tensor([1])] for k in range(len(exp_coef_structure))]
                polynomial_structure = [[TorchPolynomial(p[0])] for p in polynomial_structure]
            self.segment_functions = [
                SegmentFunction(exp_coef_structure[k], polynomial_structure[k])
                for k in range(len(exp_coef_structure))]
        else:
            self.segment_functions = segment_functions
            self.exp_coef_structure = [f.exp_coefs for f in segment_functions]
            self.polynomial_structure = [f.polynomes for f in segment_functions]

    def __add__(self, other):
        other_function = self._coerce_to_piecewise_function(other)
        assert torch.all(self.term_structure == other_function.term_structure)  # todo: at some point we could relax this
        new_segment_functions = [
            self.segment_functions[k] + other_function.segment_functions[k] for k in range(len(self.segment_functions))
        ]
        return PiecewiseFunction(self.term_structure, segment_functions=new_segment_functions)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_segment_functions = [
                f * other for f in self.segment_functions
            ]
        elif isinstance(other, PiecewiseFunction):
            assert torch.all(self.term_structure == other.term_structure)
            new_segment_functions = [
                self.segment_functions[k] * other.segment_functions[k] for k in range(len(self.segment_functions))
            ]
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return PiecewiseFunction(self.term_structure, segment_functions=new_segment_functions)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return - 1 * self.__sub__(other)

    def __pow__(self, power):
        return PiecewiseFunction(
            term_structure=self.term_structure,
            segment_functions=[f ** power for f in self.segment_functions]
        )

    def derivative(self):
        return PiecewiseFunction(
            term_structure=self.term_structure,
            segment_functions=[f.derivative() for f in self.segment_functions]
        )

    def antiderivative(self):
        return PiecewiseFunction(
            term_structure=self.term_structure,
            segment_functions=[f.antiderivative() for f in self.segment_functions]
        )

    def _truncate_term_structure(self, a: float, b: float) -> Tuple[torch.Tensor, List[int]]:
        # get term structure defined over [a, b] and corresponding mask for coefficients
        is_before_a = (self.term_structure < a)
        if not any(is_before_a):
            greatest_before_a = 0
        else:
            greatest_before_a = is_before_a.nonzero().reshape(-1)[-1]

        is_in_segment = (self.term_structure >= a) & (self.term_structure < b)

        used_term_structure = self.term_structure[is_in_segment]
        used_function_idxs = is_in_segment.nonzero().reshape(-1)

        if a not in used_term_structure:
            if a < self.term_structure[0]:
                used_term_structure[0] = a
            else:
                used_term_structure = torch.nn.functional.pad(
                    used_term_structure,
                    (1, 0),
                    "constant",
                    a
                )
                used_function_idxs = torch.nn.functional.pad(
                    used_function_idxs,
                    (1, 0),
                    "constant",
                    greatest_before_a
                )

        if b not in used_term_structure:
            used_term_structure = torch.nn.functional.pad(
                used_term_structure,
                (0, 1),
                "constant",
                b
            )
            if used_function_idxs[-1] >= len(self.segment_functions):
                used_function_idxs[-1] = len(self.segment_functions)

        return used_term_structure, used_function_idxs

    def integral(self, a, b=None):
        if b is None:
            b = a
            a = 0
        if a > b:
            return - self.integral(b, a)
        elif a == b:
            return torch.zeros(1, dtype=torch.float64).squeeze()
        else:
            term_structure, function_idxs = self._truncate_term_structure(a, b)
            antiderivatives = [self.segment_functions[k].antiderivative() for k in function_idxs]
            segment_values = [F(term_structure[k + 1]) - F(term_structure[k]) for k, F in enumerate(antiderivatives)]
            return torch.sum(torch.stack(segment_values))

    def __call__(self, x: float) -> torch.Tensor:
        is_relevant_segment = (self.term_structure <= x).nonzero()
        if not len(is_relevant_segment):
            function_idx = 0
        elif len(is_relevant_segment) == len(self.term_structure):
            function_idx = -1
        else:
            function_idx = is_relevant_segment[-1]
        return self.segment_functions[function_idx](x)

    def get_exponential_forward_integral(self, start: float = None):
        # return function that looks like f: t -> exp(Integral_start^t self(s)ds)
        if start is None:
            start = self.term_structure[0]
        new_segment_functions = []
        for k, t in enumerate(self.term_structure[:-1]):
            integral_remainder = torch.exp(self.integral(start, t))
            primitive = self.segment_functions[k].antiderivative()
            primitive_constant = primitive(t)
            function_part = (primitive - primitive_constant).get_exponential()
            new_segment_functions.append(integral_remainder * function_part)
        return PiecewiseFunction(
            term_structure=self.term_structure,
            segment_functions=new_segment_functions
        )

    def get_exponential_backward_integral(self, end: float = None):
        # return function that looks like f: t -> exp(Integral_t^end self(s)ds)
        if end is None:
            end = self.term_structure[-1]
        return (- 1 * self).get_exponential_forward_integral(start=end)

    def _coerce_to_piecewise_function(self, other):
        if isinstance(other, PiecewiseFunction):
            return other
        elif isinstance(other, int) or isinstance(other, float):
            return PiecewiseFunction(
                term_structure=self.term_structure,
                exp_coef_structure=None,
                polynomial_structure=[[torch.tensor(other)] * (len(self.term_structure))]
            )
        elif isinstance(other, torch.Tensor):
            assert len(other.reshape(-1)) == 1, "Only 1-element tensors can be converted to constant function"
            return PiecewiseFunction(
                term_structure=self.term_structure,
                exp_coef_structure=None,
                polynomial_structure=[[other.clone()]] * (len(self.term_structure))
            )
        else:
            raise TypeError(f"Unsupported type for PiecewiseFunction {type(other)}")


class OldPiecewiseIntegral:
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

        def integrate_function(a: float, b: float) -> torch.Tensor:
            used_term_structure, used_coefs, _ = self._truncate_term_structure(a, b)
            segment_integrals = used_coefs * torch.diff(used_term_structure)
            res = torch.sum(segment_integrals)
            return res

        return integrate_function

    def _integrate_exponential(self) -> Callable:

        def integrate_function(a: float, b: float) -> torch.Tensor:
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


def test_segment_function():
    case_1 = {
        "exp_coefs": torch.tensor([-1]),
        "polynomes": [torch.tensor([1, 1, 1])]
    }

    case_2 = {
        "exp_coefs": torch.tensor([-1, -1]),
        "polynomes": [torch.tensor([1, 1, 1]), torch.tensor([1, 1])]
    }

    case_2_bis = {
        "exp_coefs": torch.tensor([-1]),
        "polynomes": [torch.tensor([2, 2, 1])]
    }
    sf_2 = SegmentFunction(**case_2)
    sf_2_bis = SegmentFunction(**case_2_bis)
    assert sf_2 == sf_2_bis
    sf = SegmentFunction(**case_1)
    assert sf(0) == 1
    assert sf(1).float() == 3 * torch.exp(torch.tensor(-1))
    sf_d = sf.derivative()
    sf_d_test = SegmentFunction(torch.tensor([-1]), torch.tensor([-1, -1, -1]) + torch.tensor([1, 2, 0]))
    print(sf.antiderivative())
    print("Segment function tests passed")
    return


def test_old_constant():
    case_1 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.ones(4, dtype=torch.float64)
    }
    case_2 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "coef_structure": torch.arange(4, dtype=torch.float64)
    }

    integral_1 = OldPiecewiseIntegral(**case_1)
    assert integral_1(0, 1).item() == 1
    assert integral_1(0, 2.5).item() == 2.5
    assert integral_1(-1, 2).item() == 3
    assert integral_1(0, 6).item() == 6

    integral_2 = OldPiecewiseIntegral(**case_2)
    assert integral_2(0, 1).item() == 0
    assert integral_2(0, 2.5).item() == 2
    assert integral_2(-1, 2).item() == 1
    assert integral_2(3, 6).item() == 9

    print('All constant test cases passed')
    return


def test_old_exponential():
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
    integral_1 = OldPiecewiseIntegral(**case_1)
    assert integral_1(0, 1).item() == np.exp(1) - 1
    assert integral_1(0, 4.5).item() == np.exp(4.5) - 1

    integral_2 = OldPiecewiseIntegral(**case_2)
    assert integral_2(0, 1).item() == 0
    assert integral_2(0, 2).item() == np.exp(2) - np.exp(1)
    assert integral_2(0, 3).item() == (np.exp(2) - np.exp(1)) + 2 * (np.exp(3 * 2) - np.exp(2 * 2)) / 2

    print("All exponential test cases passed")
    return


def test_constant():
    case_1 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "exp_coef_structure": torch.zeros(5, dtype=torch.float64),
        "polynomial_structure": [torch.ones(1, dtype=torch.float64)] * 5
    }
    case_2 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "exp_coef_structure": torch.zeros(5, dtype=torch.float64),
        "polynomial_structure": [torch.ones(1, dtype=torch.float64) * k for k in range(5)]
    }

    function_1 = PiecewiseFunction(**case_1)
    assert function_1.integral(0, 1).item() == 1
    assert function_1.integral(0, 2.5).item() == 2.5
    assert function_1.integral(-1, 2).item() == 3
    assert function_1.integral(0, 6).item() == 6

    assert (function_1 ** 5).integral(0, 1).item() == 1

    function_2 = PiecewiseFunction(**case_2)
    assert function_2.integral(0, 1).item() == 0
    assert function_2.integral(0, 2.5).item() == 2
    assert function_2.integral(-1, 2).item() == 1
    assert function_2.integral(3, 6).item() == 11

    print('All constant test cases passed')
    return


def test_exponential():
    case_1 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "exp_coef_structure": torch.ones(5, dtype=torch.float64),
        "polynomial_structure": [torch.ones(1, dtype=torch.float64)] * 5
    }
    case_2 = {
        "term_structure": torch.arange(5, dtype=torch.float64),
        "exp_coef_structure": torch.arange(5, dtype=torch.float64),
        "polynomial_structure": [torch.ones(1) * k for k in range(5)]
    }
    function_1 = PiecewiseFunction(**case_1)
    assert function_1.integral(0, 1).item() == np.exp(1) - 1
    assert function_1.integral(0, 4.5).item() == np.exp(4.5) - 1

    function_2 = PiecewiseFunction(**case_2)
    assert function_2.integral(0, 1).item() == 0
    assert function_2.integral(0, 2).item() == np.exp(2) - np.exp(1)
    assert function_2.integral(0, 3).item() == (np.exp(2) - np.exp(1)) + 2 * (np.exp(3 * 2) - np.exp(2 * 2)) / 2

    print("All exponential test cases passed")
    return


if __name__ == "__main__":
    test_polynomes()
    test_segment_function()
    test_old_constant()
    test_old_exponential()
    test_constant()
    test_exponential()
