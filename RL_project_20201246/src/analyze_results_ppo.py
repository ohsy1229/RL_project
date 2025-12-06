import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.portfolio_env_continuous import PortfolioEnvContinuous
from utils.baselines import equal_weight_returns, buy_and_hold_returns
from utils.metrics import nav_from_returns


def run_policy_on_env_nav(model, env: PortfolioEnvContinuous) -> pd.Series:
    """
    정책 실행 후 NAV 시계열을 반환하는 함수입니다.
    env.step()에서 NAV는 net_ret(거래비용 반영)을 기반으로 업데이트됩니다.
    """
    obs, info = env.reset()
    navs = [info.get("nav", 1.0)]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if "nav" in info:
            navs.append(info["nav"])

        if terminated or truncated:
            break

    return pd.Series(navs)


def main():
    data_path = os.path.join("data", "processed", "prices.csv")

    # 연속 액션 PPO 결과는 별도 폴더에 저장했다고 가정합니다.
    results_dir = os.path.join("results_continuous")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    figs_dir = os.path.join(results_dir, "figures")

    os.makedirs(figs_dir, exist_ok=True)

    # 데이터 로드
    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    asset_cols = ["SEC", "HYU", "NAVER"]

    train_df = price_df.loc[:"2016-12-30"]
    valid_df = price_df.loc["2017-01-02":"2019-12-30"]
    test_df = price_df.loc["2020-01-02":]

    # -------------------------------
    # 1) Validation NAV Paths by Seed
    # -------------------------------
    seeds = [0, 42, 2024]
    nav_paths = {}

    for seed in seeds:
        model_path = os.path.join(models_dir, f"ppo_cont_seed_{seed}.zip")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}, skip.")
            continue

        model = PPO.load(model_path)

        env_valid = PortfolioEnvContinuous(
            price_df=valid_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=0.3,
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        navs = run_policy_on_env_nav(model, env_valid)
        nav_paths[seed] = navs

    if nav_paths:
        plt.figure(figsize=(10, 5))
        for seed, navs in nav_paths.items():
            plt.plot(navs.values, label=f"seed {seed}")
        plt.title("Validation NAV Paths by Seed (PPO Continuous)")
        plt.xlabel("Time (validation period)")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(figs_dir, "nav_valid_by_seed_ppo_cont.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {fig_path}")

    # -----------------------------------------
    # 2) Test 구간 RL(PPO) vs Baseline NAV 비교
    # -----------------------------------------
    best_model_path = os.path.join(models_dir, "ppo_cont_best.zip")
    if os.path.exists(best_model_path):
        model_best = PPO.load(best_model_path)

        env_test = PortfolioEnvContinuous(
            price_df=test_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=0.3,
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        nav_rl = run_policy_on_env_nav(model_best, env_test)

        # Baseline: EW (월별 리밸런싱 + 비용), BH (리밸 X, 비용 0)
        ew_ret = equal_weight_returns(
            test_df, asset_cols, trading_cost=0.002, rebalance_freq="M"
        )
        bh_ret = buy_and_hold_returns(test_df, asset_cols)

        nav_ew = nav_from_returns(ew_ret)
        nav_bh = nav_from_returns(bh_ret)

        # 길이 맞추기
        min_len = min(len(nav_rl), len(nav_ew), len(nav_bh))
        x = range(min_len)

        plt.figure(figsize=(10, 5))
        plt.plot(x, nav_rl.values[:min_len], label="RL (PPO, continuous)")
        plt.plot(x, nav_ew.values[:min_len], label="Equal Weight (net, monthly rebalance)")
        plt.plot(x, nav_bh.values[:min_len], label="Buy & Hold")
        plt.title("Test NAV Comparison: PPO Continuous RL vs Baselines")
        plt.xlabel("Time (test period)")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(figs_dir, "nav_test_comparison_ppo_cont.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {fig_path}")
    else:
        print(f"Best PPO model not found at {best_model_path}, skip test NAV plot.")


if __name__ == "__main__":
    main()
