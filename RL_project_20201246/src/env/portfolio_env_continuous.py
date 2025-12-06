import numpy as np
import pandas as pd
from typing import List, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnvContinuous(gym.Env):
    """
    Continuous Trading Environment for 3-stock Portfolio.

    Action (continuous):
        a_t ∈ [-1, 1]^3
        -> target weights in [0, max_weight], then normalized to sum to 1.

    Reward:
        r_t = log(1 + net_ret_t) - dd_penalty * DD_t - turn_penalty * turnover_t

        net_ret_t = w_t · ret_{t+1} - trading_cost * turnover_t
        turnover_t = sum_i |w_t(i) - w_{t-1}(i)|

    Observation:
        - window_size 일간 각 종목 일일 수익률 (window_size * 3)
        - 현재 포트폴리오 비중 (3)
        - 각 종목별 20일 모멘텀, 60일 모멘텀, 20일 변동성 (3 * 3 = 9)
        - 시장 지수(옵션): 20일 모멘텀, 60일 모멘텀, 20일 변동성 (3)
          (market_col이 없으면 0으로 채움)
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
        turn_penalty: float = 0.001,
        max_weight: float = 0.5,
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
        self.turn_penalty = turn_penalty
        self.max_weight = max_weight
        self.initial_nav = initial_nav

        # --- 기본 수익률 계산 ---
        self.ret_df = self.price_df[self.asset_cols].pct_change().fillna(0.0)

        # --- feature engineering: 자산별 모멘텀 / 변동성 ---
        # 20일 / 60일 모멘텀 (단순 합), 20일 변동성
        self.mom20_df = self.ret_df.rolling(20).sum().fillna(0.0)
        self.mom60_df = self.ret_df.rolling(60).sum().fillna(0.0)
        self.vol20_df = self.ret_df.rolling(20).std().fillna(0.0)

        # --- market 지수가 있을 경우, 같은 feature 생성 ---
        if self.market_col is not None and self.market_col in self.price_df.columns:
            m_ret = self.price_df[self.market_col].pct_change().fillna(0.0)
            self.market_mom20 = m_ret.rolling(20).sum().fillna(0.0)
            self.market_mom60 = m_ret.rolling(60).sum().fillna(0.0)
            self.market_vol20 = m_ret.rolling(20).std().fillna(0.0)
        else:
            self.market_mom20 = None
            self.market_mom60 = None
            self.market_vol20 = None

        # --- Action space (continuous) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(asset_cols),), dtype=np.float32
        )

        # --- Observation space dimension 계산 ---
        n_assets = len(asset_cols)
        base_ret_dim = window_size * n_assets
        weight_dim = n_assets
        asset_feat_dim = n_assets * 3  # (mom20, mom60, vol20) per asset
        market_feat_dim = 3  # (mom20, mom60, vol20) for market (없어도 0으로 채움)

        obs_dim = base_ret_dim + weight_dim + asset_feat_dim + market_feat_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_idx: int = None
        self.weights: np.ndarray = None
        self.nav: float = None
        self.peak_nav: float = None

    # ------------------------
    # Helper
    # ------------------------
    def _get_observation(self) -> np.ndarray:
        
        # 1) window_size 동안의 일일 수익률 (자산별)
        start = self.current_idx - self.window_size + 1
        window_returns = self.ret_df.iloc[start : self.current_idx + 1]
        ret_part = window_returns.to_numpy().flatten()

        # 2) 현재 weights
        w_part = self.weights

        # 3) 자산별 모멘텀/변동성 (현재 시점 기준)
        idx = self.ret_df.index[self.current_idx]
        mom20 = self.mom20_df.loc[idx, self.asset_cols].to_numpy(dtype=float)
        mom60 = self.mom60_df.loc[idx, self.asset_cols].to_numpy(dtype=float)
        vol20 = self.vol20_df.loc[idx, self.asset_cols].to_numpy(dtype=float)
        asset_feat = np.concatenate([mom20, mom60, vol20])

        # 4) 시장 지수 feature (없으면 0으로 채움)
        if self.market_mom20 is not None:
            m_mom20 = float(self.market_mom20.loc[idx])
            m_mom60 = float(self.market_mom60.loc[idx])
            m_vol20 = float(self.market_vol20.loc[idx]) if not np.isnan(self.market_vol20.loc[idx]) else 0.0
            market_feat = np.array([m_mom20, m_mom60, m_vol20], dtype=float)
        else:
            market_feat = np.zeros(3, dtype=float)

        obs = np.concatenate([ret_part, w_part, asset_feat, market_feat])
        return obs.astype(np.float32)

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        a ∈ [-1, 1]^3  ->  target weights in [0, max_weight], sum to 1.
        """
        a = np.clip(action, -1.0, 1.0).astype(float)

        # [-1,1] -> [0,1] -> [0, max_weight]
        w_raw = (a + 1.0) / 2.0 * self.max_weight

        s = w_raw.sum()
        if s > 0:
            w = w_raw / s
        else:
            w = np.ones_like(w_raw) / len(w_raw)
        return w

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_idx = self.window_size
        self.weights = np.array([1/3, 1/3, 1/3], dtype=float)
        self.nav = float(self.initial_nav)
        self.peak_nav = float(self.initial_nav)

        obs = self._get_observation()
        info = {"nav": self.nav}
        return obs, info

    def step(self, action: np.ndarray):
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        old_weights = self.weights.copy()
        new_weights = self._action_to_weights(action)

        # 데이터 끝에 도달하면 종료
        if self.current_idx + 1 >= len(self.ret_df):
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, info

        # 다음날 수익률
        ret_row = self.ret_df.iloc[self.current_idx + 1]
        asset_returns = ret_row.to_numpy(dtype=float)

        # 포트폴리오 수익률 (새 비중 기준)
        gross_ret = float(np.dot(new_weights, asset_returns))

        # turnover / 거래비용
        turnover = float(np.sum(np.abs(new_weights - old_weights)))
        cost = self.trading_cost * turnover

        net_ret = gross_ret - cost

        # NAV 업데이트 (net_ret 기준)
        new_nav = self.nav * (1.0 + net_ret)

        # Drawdown
        self.peak_nav = max(self.peak_nav, new_nav)
        dd = (self.peak_nav - new_nav) / self.peak_nav

        # 위험 패널티
        dd_term = self.dd_penalty * dd
        turn_term = self.turn_penalty * turnover

        # 보상: log wealth change - risk penalties
        safe_net_ret = max(net_ret, -0.99)
        reward_wealth = np.log(1.0 + safe_net_ret)
        reward = reward_wealth - dd_term - turn_term

        # 상태 업데이트
        self.nav = new_nav
        self.weights = new_weights
        self.current_idx += 1

        obs = self._get_observation()
        info = {
            "nav": self.nav,
            "port_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "drawdown": dd,
            "weights": self.weights.copy(),
        }
        return obs, reward, terminated, truncated, info
