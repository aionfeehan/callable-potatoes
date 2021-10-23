from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import torch
import uvicorn


from LGM import LGM1F
from piecewise_functions import PiecewiseFunction


class Option(BaseModel):
    expiry: float
    maturity: float
    strike: float


def build_test_lgm():
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
        'sigma_structure': np.ones(5) * 1e-2,
        'n_paths': 10000
    }

    lgm = LGM1F(**case_1)
    return lgm


lgm_model = build_test_lgm()
app = FastAPI()


@app.get("/ping")
async def get_ping():
    return {"message": "received get ping"}


@app.post("/price")
async def get_price(option: Option):
    print(option)
    greeks = lgm_model.get_caplet_greeks(option.expiry, option.maturity, option.strike)
    print(greeks)
    res = {
        'pv': greeks['pv'],
        'vega': greeks['dsigma'],
        'delta': greeks['dm']
    }
    return res


if __name__ == '__main__':
    uvicorn.run("lgm_service:app", host='localhost', port=8081, log_level="info")
