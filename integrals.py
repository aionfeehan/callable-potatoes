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
            constant_structure: torch.Tensor = None
    ):
        assert len(term_structure) == len(coef_structure) + 1
        assert constant_structure is None or (len(constant_structure) == len(coef_structure))
        self.term_structure = term_structure
        self.coef_structure = coef_structure
        self.constant_structure = constant_structure
        self._constant_integral_function = self._integrate_constant()
        self._exponential_integral_function = self._integrate_exponential()

    def __call__(self, a, b=None, _type='constant'):
        if b is None:
            b = a
            a =  0
        if _type == 'constant':
            return self._constant_integral_function(a, b)
        elif _type == 'exponential':
            return self._exponential_integral_function(a, b)

    def _integrate_constant(self) -> Callable:

        def integrate_function(a: float, b: float) -> float:
            used_term_structure, used_coefs, _ = self._truncate_term_structure(a, b)
            segment_integrals = used_coefs * torch.diff(used_term_structure)
            res = torch.sum(segment_integrals).item()
            return res

        return integrate_function

    def _integrate_exponential(self) -> Callable:

        def integrate_function(a: float, b: float) -> float:
            used_term_structure, used_coefs, used_constant_structure = self._truncate_term_structure(a, b)
            if used_constant_structure is None:
                used_constant_structure = 1
            segment_end_values = torch.exp(used_coefs * used_term_structure[1:]) / used_coefs
            segment_start_values = torch.exp(used_coefs * used_term_structure[:-1]) / used_coefs
            segment_integrals = (segment_end_values - segment_start_values) * used_constant_structure
            res = torch.sum(segment_integrals)
            return res

        return integrate_function

    def _truncate_term_structure(self, a: float, b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get term structure defined over [a, b] and corresponding mask for coefficients
        is_in_segment = (self.term_structure >= a) & (self.term_structure <= b)
        used_term_structure = self.term_structure[is_in_segment]
        used_coefs = self.coef_structure[is_in_segment[:-1]]
        if self.constant_structure:
            used_constants = self.constant_structure[is_in_segment[:-1]]
        else:
            used_constants = None

        if a not in self.term_structure:
            used_term_structure = torch.nn.functional.pad(used_term_structure, (1, 0), "constant", a)
            if a < self.term_structure[0]:
                used_coefs = torch.nn.functional.pad(used_coefs, (1,  0), "constant", used_coefs[0])
                if used_constants:
                    used_constants = torch.nn.functional.pad(used_constants, (1, 0), "constant", used_constants[0])

        if b not in self.term_structure:
            used_term_structure = torch.nn.funcitonal.pad(used_term_structure, (0, 1), "constant", b)
            if b > self.term_structure[-1]:
                used_coefs = torch.nn.functional.pad(used_coefs, (1, 0), "constant", used_coefs[-1])
                if used_constants:
                    used_constants = torch.nn.functional.pad(used_constants, (0, 1), "constant", used_constants[-1])

        return used_term_structure, used_coefs, used_constants


