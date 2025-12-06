import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, List
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnvDiscrete(gym.Env):
    """
    Discrete Trading Environment for 3-stock Portfolio.
    Actions:
        0: Hold
        1: Buy SEC
        2: Sell SEC
        3: Buy HYU
        4: Sell HYU
        5: Buy NAVER
        6: Sell NAVER
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_df: pd.DataFrame,
        asset_cols: List[str],
        market_col: str = None,
        window_size: int = 20,
        trading_cost: float = 0.002,
        dd_penalty: float = 0.5,
        trade_frac: float = 0.1,
        max_weight: float = 0.5,
        turn_penalty: float = 0.0,
        initial_nav: float = 1.0,
    ):
        super().__init__()

        assert len(asset_cols) == 3, "This env currently assumes exactly 3 assets."

        self.price_df = price_df.copy()
        self.asset_cols = asset_cols
        self.market_col = market_col
        self.window_size = window_size
        self.trading_cost = trading_cost
        self.dd_penalty = dd_penalty
        self.trade_frac = trade_frac
        self.max_weight = max_weight
        self.turn_penalty = turn_penalty
        self.initial_nav = initial_nav

        # Return 데이터 미리 계산
        self.ret_df = self.price_df.pct_change().fillna(0.0)

        # Action space: 7 discrete
        self.action_space = spaces.Discrete(7)

        # Observation space:
        #   past returns (window_size * n_assets) + current weights (n_assets)
        obs_dim = window_size * len(asset_cols) + len(asset_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_idx: int = None
        self.weights = None
        self.nav = None
        self.peak_nav = None

    # ------------------------
    # Helper functions
    # ------------------------
    def _get_observation(self) -> np.ndarray:
        """Flattened past returns + current weights"""
        start = self.current_idx - self.window_size + 1
        window_returns = self.ret_df[self.asset_cols].iloc[start : self.current_idx + 1]
        obs = np.concatenate(
            [
                window_returns.to_numpy().flatten(),
                self.weights,
            ]
        )
        return obs.astype(np.float32)

    def _apply_action(self, action: int) -> np.ndarray:
        """Update weights according to discrete action."""
        new_w = self.weights.copy()

        if action == 1:  # Buy SEC
            new_w[0] += self.trade_frac
        elif action == 2:  # Sell SEC
            new_w[0] -= self.trade_frac
        elif action == 3:  # Buy HYU
            new_w[1] += self.trade_frac
        elif action == 4:  # Sell HYU
            new_w[1] -= self.trade_frac
        elif action == 5:  # Buy NAVER
            new_w[2] += self.trade_frac
        elif action == 6:  # Sell NAVER
            new_w[2] -= self.trade_frac
        # else (0: Hold): no change

        # Clip to [0, max_weight]
        new_w = np.clip(new_w, 0.0, self.max_weight)

        # Normalize to sum to 1 (항상 풀 투자 구조 유지)
        s = new_w.sum()
        if s > 0:
            new_w = new_w / s
        else:
            new_w[:] = 1.0 / len(new_w)

        return new_w

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # window_size 만큼의 과거 수익률이 보이도록 시작 인덱스 설정
        self.current_idx = self.window_size
        self.weights = np.array([1/3, 1/3, 1/3], dtype=float)
        self.nav = float(self.initial_nav)
        self.peak_nav = float(self.initial_nav)

        obs = self._get_observation()
        info = {"nav": self.nav}
        return obs, info

    def step(self, action: int):
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        old_weights = self.weights.copy()
        new_weights = self._apply_action(int(action))

        # End of data: 더 이상 다음날 수익률이 없으면 종료
        if self.current_idx + 1 >= len(self.ret_df):
            terminated = True
            # 마지막 obs는 현재 상태 기준으로 반환
            return self._get_observation(), 0.0, terminated, truncated, info

        # Next-day returns
        ret_row = self.ret_df.iloc[self.current_idx + 1]
        asset_returns = ret_row[self.asset_cols].to_numpy(dtype=float)

        # Portfolio return (액션으로 만든 new_weights 기준으로 계산)
        gross_ret = float(np.dot(new_weights, asset_returns))

        # Trading cost (리밸런싱 전후 비중 차이 기준)
        turnover = float(np.sum(np.abs(new_weights - old_weights)))
        cost = self.trading_cost * turnover

        # Net return: 비용 반영
        net_ret = gross_ret - cost

        # NAV update는 net_ret 기준
        new_nav = self.nav * (1.0 + net_ret)

        # Drawdown 계산 (NAV 기준, penalty는 reward에서만 사용)
        self.peak_nav = max(self.peak_nav, new_nav)
        dd = (self.peak_nav - new_nav) / self.peak_nav
        dd_pen = self.dd_penalty * dd

        # Turnover penalty (선택적 리스크 패널티)
        extra_turn = self.turn_penalty * turnover

        # Reward: net_ret에서 리스크 페널티를 차감
        reward = net_ret - extra_turn - dd_pen

        # Update environment state
        self.nav = new_nav
        self.weights = new_weights
        self.current_idx += 1

        obs = self._get_observation()
        info = {
            "nav": self.nav,
            "port_ret": gross_ret,   # 비용 전 수익률
            "net_ret": net_ret,      # 비용 반영 후 수익률
            "turnover": turnover,
            "drawdown": dd,
            "weights": self.weights.copy(),
        }

        return obs, reward, terminated, truncated, info
