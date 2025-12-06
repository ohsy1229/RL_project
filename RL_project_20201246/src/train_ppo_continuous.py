import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.portfolio_env_continuous import PortfolioEnvContinuous
from utils.metrics import performance_stats
from utils.baselines import equal_weight_returns, buy_and_hold_returns


def run_policy_on_env(model, env: PortfolioEnvContinuous) -> pd.Series:
    """
    학습된 정책을 환경에서 한 번 전체 기간 실행하고,
    일별 net_ret 시계열을 반환한다.
    """
    obs, info = env.reset()
    rets = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if "net_ret" in info:
            rets.append(info["net_ret"])

        if terminated or truncated:
            break

    return pd.Series(rets)


def main():
    data_path = os.path.join("data", "processed", "prices.csv")
    results_dir = os.path.join("results_continuous")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 데이터
    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    asset_cols = ["SEC", "HYU", "NAVER"]

    train_df = price_df.loc[:"2016-12-30"]
    valid_df = price_df.loc["2017-01-02":"2019-12-30"]
    test_df = price_df.loc["2020-01-02":]

    print("Train length:", len(train_df), "Valid length:", len(valid_df))

    seeds = [0, 42, 2024]
    results_valid = []

    # --------- 1) seed별 Train / Valid ---------
    for seed in seeds:
        print(f"\n=== Training PPO (continuous) with seed {seed} ===")

        env_train = PortfolioEnvContinuous(
            price_df=train_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=0.3,     # log reward라서 약간 줄여도 됨
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env_train,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            seed=seed,
        )

        model.learn(total_timesteps=150_000)

        model_path = os.path.join(models_dir, f"ppo_cont_seed_{seed}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # Validation 평가
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

        ret_valid = run_policy_on_env(model, env_valid)
        stats_valid = performance_stats(ret_valid)
        stats_valid["seed"] = seed
        print("Validation stats:", stats_valid)
        results_valid.append(stats_valid)

    df_valid = pd.DataFrame(results_valid).set_index("seed")
    print("\nValidation performance (PPO, continuous):")
    print(df_valid)

    valid_log_path = os.path.join(logs_dir, "validation_results_ppo.csv")
    df_valid.to_csv(valid_log_path)
    print(f"\nValidation results saved to {valid_log_path}")

    # --------- 2) best seed로 Train+Valid 재학습, Test 평가 ---------
    best_seed = df_valid["Sharpe"].idxmax()
    print(f"\nBest seed (by Sharpe): {best_seed}")

    train_full_df = pd.concat([train_df, valid_df])

    env_train_full = PortfolioEnvContinuous(
        price_df=train_full_df,
        asset_cols=asset_cols,
        window_size=20,
        trading_cost=0.002,
        dd_penalty=0.3,
        turn_penalty=0.0005,
        max_weight=0.5,
        initial_nav=1.0,
    )

    model_best = PPO(
        policy="MlpPolicy",
        env=env_train_full,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=int(best_seed),
    )

    model_best.learn(total_timesteps=200_000)

    best_model_path = os.path.join(models_dir, "ppo_cont_best.zip")
    model_best.save(best_model_path)
    print(f"\nBest PPO model saved to {best_model_path}")

    # Test 평가
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

    ret_test_rl = run_policy_on_env(model_best, env_test)
    rl_stats = performance_stats(ret_test_rl)
    rl_stats["Strategy"] = "RL_PPO_CONT"

    # Baseline (동일 비용 구조)
    ew_ret = equal_weight_returns(test_df, asset_cols, trading_cost=0.002, rebalance_freq="M")
    bh_ret = buy_and_hold_returns(test_df, asset_cols)

    ew_stats = performance_stats(ew_ret)
    ew_stats["Strategy"] = "EW"

    bh_stats = performance_stats(bh_ret)
    bh_stats["Strategy"] = "BH"

    df_test = pd.DataFrame([rl_stats, ew_stats, bh_stats]).set_index("Strategy")
    print("\nTest performance (PPO continuous RL vs Baselines):")
    print(df_test)

    test_log_path = os.path.join(logs_dir, "test_results_ppo.csv")
    df_test.to_csv(test_log_path)
    print(f"\nTest results saved to {test_log_path}")

    # NAV 시계열 저장
    nav_rl = (1 + ret_test_rl.fillna(0.0)).cumprod()
    nav_rl.name = "RL_PPO_CONT_NAV"
    nav_path = os.path.join(logs_dir, "test_nav_rl_ppo_cont.csv")
    nav_rl.to_csv(nav_path)
    print(f"RL NAV (Test, PPO continuous) saved to {nav_path}")


if __name__ == "__main__":
    main()
