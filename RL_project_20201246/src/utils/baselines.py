import numpy as np
import pandas as pd
from typing import List


def equal_weight_returns(
    price_df: pd.DataFrame,
    asset_cols: List[str],
    trading_cost: float = 0.0,
    rebalance_freq: str = "M",  # "D"=매일, "M"=매월
) -> pd.Series:
    """
    Equal-Weight 포트폴리오의 일별 '순수익률(net_ret)' 시계열을 계산한다.

    - RL 환경과 동일한 거래비용 구조 사용:
        net_ret_t = dot(w_t, r_t) - trading_cost * turnover_t
        turnover_t = sum(|w_t - w_{t-1}|)

    - rebalance_freq="M": 월 단위로 equal weight로 리밸런싱
    - rebalance_freq="D": 매일 equal weight 리밸런싱 (참고용)

    초기 매수 시점에는 RL과 마찬가지로 거래비용을 부과하지 않는다
    (w_0를 이미 equal weight라고 가정).
    """
    rets = price_df[asset_cols].pct_change().fillna(0.0)
    dates = rets.index
    n = len(asset_cols)

    # 초기 비중: 이미 equal-weight로 들고 있다고 가정 → 초기 turnover=0
    prev_weights = np.ones(n, dtype=float) / n
    out = []

    # 월별 리밸런싱 여부 판별용
    prev_month = dates[0].month

    for i, (dt, r_row) in enumerate(rets.iterrows()):
        r = r_row.to_numpy(dtype=float)

        # 리밸런싱 대상 비중 결정
        new_weights = prev_weights.copy()

        if rebalance_freq == "D":
            if i > 0:  # 첫 날은 이미 equal-weight 상태이므로 리밸 X
                new_weights[:] = 1.0 / n
        elif rebalance_freq == "M":
            # 월이 바뀌는 첫 날에만 리밸런싱
            if i > 0 and dt.month != prev_month:
                new_weights[:] = 1.0 / n
            prev_month = dt.month
        else:
            raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")

        # 포트폴리오 수익률 (리밸 후 비중 기준)
        gross_ret = float(np.dot(new_weights, r))

        # RL 환경과 동일한 turnover / 비용
        turnover = float(np.sum(np.abs(new_weights - prev_weights)))
        cost = trading_cost * turnover

        net_ret = gross_ret - cost
        out.append(net_ret)

        prev_weights = new_weights

    return pd.Series(out, index=dates, name="EW_net_ret")


def buy_and_hold_returns(
    price_df: pd.DataFrame,
    asset_cols: List[str],
) -> pd.Series:
    """
    Buy & Hold 전략의 일별 수익률.

    - 초기 시점에 equal-weight로 매수 후, 추가 리밸런싱 없음
    - 따라서 turnover = 0, 거래비용 = 0 (RL env에서도 reset 시점엔 비용 없음)

    단, 비중은 가격 변화에 따라 매일 업데이트된다.
    """
    rets = price_df[asset_cols].pct_change().fillna(0.0)
    dates = rets.index
    n = len(asset_cols)

    # 초기 비중 equal-weight
    weights = np.ones(n, dtype=float) / n
    out = []

    for _, r_row in rets.iterrows():
        r = r_row.to_numpy(dtype=float)

        # 오늘 포트폴리오 수익률 (전일 비중 기준)
        gross_ret = float(np.dot(weights, r))
        out.append(gross_ret)

        # 내일 비중 업데이트 (리밸 없으므로 단순 drift)
        # 각 자산 가치: w_i * (1 + r_i)
        new_values = weights * (1.0 + r)
        total = new_values.sum()
        if total > 0:
            weights = new_values / total
        else:
            # 극단적 경우 방어용
            weights = np.ones(n, dtype=float) / n

    return pd.Series(out, index=dates, name="BH_ret")
