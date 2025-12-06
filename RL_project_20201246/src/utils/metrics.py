import numpy as np
import pandas as pd


def performance_stats(ret: pd.Series) -> dict:
    """
    일간 수익률 시계열에서 기본 성과지표 계산 함수이다.
    - CAGR
    - Annualized Vol
    - Sharpe
    - MDD
    """
    r = ret.dropna()
    if len(r) == 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MDD": np.nan}

    cumulative = (1 + r).prod()
    years = len(r) / 252
    cagr = cumulative ** (1 / years) - 1 if years > 0 else np.nan

    vol_d = r.std()
    vol_a = vol_d * np.sqrt(252) if vol_d > 0 else np.nan

    sharpe = (r.mean() / vol_d) * np.sqrt(252) if vol_d > 0 else np.nan

    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = (nav - peak) / peak
    mdd = dd.min() if len(dd) > 0 else np.nan  # 음수

    return {"CAGR": cagr, "Vol": vol_a, "Sharpe": sharpe, "MDD": mdd}


def nav_from_returns(ret: pd.Series, initial_nav: float = 1.0) -> pd.Series:
    """
    일간 수익률 시계열에서 NAV 시계열을 계산하는 함수이다.
    """
    r = ret.fillna(0.0)
    nav = (1 + r).cumprod() * initial_nav
    return nav
