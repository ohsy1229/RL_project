import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.portfolio_env_continuous import PortfolioEnvContinuous
from utils.baselines import equal_weight_returns, buy_and_hold_returns
from utils.metrics import nav_from_returns


plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True


def run_policy_on_env_nav(model, env: PortfolioEnvContinuous) -> pd.Series:
    """
    PPO 정책을 환경에서 한 번 전체 구간 실행하고,
    NAV 시계열을 반환하는 함수이다.
    info["nav"]에는 거래비용을 반영한 NAV가 저장된다.
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

    return pd.Series(navs, name="NAV")


def main():
    # ----- 경로 설정 -----
    data_path = os.path.join("data", "processed", "prices.csv")

    results_dir = os.path.join("results_continuous")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    figs_dir = os.path.join(results_dir, "figures")

    os.makedirs(figs_dir, exist_ok=True)

    # ----- 데이터 로드 및 Train/Valid/Test 분할 -----
    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    asset_cols = ["SEC", "HYU", "NAVER"]

    train_df = price_df.loc[:"2016-12-30"]
    valid_df = price_df.loc["2017-01-02":"2019-12-30"]
    test_df = price_df.loc["2020-01-02":]

    print("Train :", train_df.index[0], "->", train_df.index[-1], ",", len(train_df))
    print("Valid :", valid_df.index[0], "->", valid_df.index[-1], ",", len(valid_df))
    print("Test  :", test_df.index[0], "->", test_df.index[-1], ",", len(test_df))

    # ============================================================
    # 1) seed별 PPO 모델의 Validation NAV 경로 시각화
    # ============================================================
    seeds = [0, 42, 2024]
    nav_paths = {}

    for seed in seeds:
        model_path = os.path.join(models_dir, f"ppo_cont_seed_{seed}.zip")
        if not os.path.exists(model_path):
            print(f"[WARN] PPO model not found: {model_path} (skip)")
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
        plt.title("Validation NAV Paths by Seed (PPO, continuous)")
        plt.xlabel("Time (validation period)")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(figs_dir, "ppo_nav_valid_by_seed.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved figure: {fig_path}")

    # ============================================================
    # 2) Best PPO 모델의 Test NAV vs Baselines
    # ============================================================
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

        # Baseline: EW는 월별 리밸런싱 + 동일 거래비용, BH는 단순 buy&hold
        ew_ret = equal_weight_returns(
            test_df, asset_cols, trading_cost=0.002, rebalance_freq="M"
        )
        bh_ret = buy_and_hold_returns(test_df, asset_cols)

        nav_ew = nav_from_returns(ew_ret)
        nav_bh = nav_from_returns(bh_ret)

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

        fig_path = os.path.join(figs_dir, "ppo_nav_test_comparison.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved figure: {fig_path}")
    else:
        print(f"[WARN] Best PPO model not found at {best_model_path}, skip test NAV plot.")

    # ============================================================
    # 3) 하이퍼파라미터 실험 결과 (validation / test) 로드 및 출력
    # ============================================================
    hp_valid_path = os.path.join(logs_dir, "validation_results_ppo_hparam.csv")
    hp_test_path = os.path.join(logs_dir, "test_results_ppo_hparam.csv")

    if os.path.exists(hp_valid_path):
        df_valid_hp = pd.read_csv(hp_valid_path, index_col=0)
        print("\n[Hyperparameter] PPO Validation results:")
        print(df_valid_hp[["learning_rate", "dd_penalty", "CAGR", "Vol", "Sharpe", "MDD"]])

        # Sharpe 기준 막대그래프
        plt.figure(figsize=(8, 4))
        df_valid_hp["Sharpe"].plot(kind="bar")
        plt.title("PPO Hyperparameter Comparison (Validation Sharpe)")
        plt.ylabel("Sharpe ratio")
        plt.xlabel("Config")
        plt.grid(True, axis="y", alpha=0.3)

        fig_path = os.path.join(figs_dir, "ppo_hparam_validation_sharpe.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved figure: {fig_path}")
    else:
        print(f"[WARN] Hyperparameter validation results not found at {hp_valid_path}")

    if os.path.exists(hp_test_path):
        df_test_hp = pd.read_csv(hp_test_path, index_col=0)
        print("\n[Hyperparameter] PPO Test results:")
        print(df_test_hp[["learning_rate", "dd_penalty", "CAGR", "Vol", "Sharpe", "MDD"]])
    else:
        print(f"[WARN] Hyperparameter test results not found at {hp_test_path}")


if __name__ == "__main__":
    main()
