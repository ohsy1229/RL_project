# src/train_ppo_continuous_hparam.py

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.portfolio_env_continuous import PortfolioEnvContinuous
from utils.metrics import performance_stats
from utils.baselines import equal_weight_returns, buy_and_hold_returns


def run_policy_on_env_returns(model, env: PortfolioEnvContinuous) -> pd.Series:
    """
    학습된 PPO 정책을 env에서 한 번 전체 구간 실행하고,
    net_ret(거래비용 반영) 시계열을 반환한다.
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
    # ----- 경로 설정 -----
    data_path = os.path.join("data", "processed", "prices.csv")
    results_dir = os.path.join("results_continuous")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ----- 데이터 로드 -----
    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    asset_cols = ["SEC", "HYU", "NAVER"]

    train_df = price_df.loc[:"2016-12-30"]
    valid_df = price_df.loc["2017-01-02":"2019-12-30"]
    test_df  = price_df.loc["2020-01-02":]

    print("Train length:", len(train_df),
          "Valid length:", len(valid_df),
          "Test length:", len(test_df))

    # ----- 하이퍼파라미터 설정 목록 -----
    configs = [
        {
            "name": "base_lr1e-4_dd0.3",
            "learning_rate": 1e-4,
            "dd_penalty": 0.3,
        },
        {
            "name": "lr3e-4_dd0.3",
            "learning_rate": 3e-4,
            "dd_penalty": 0.3,
        },
        {
            "name": "lr1e-4_dd0.1",
            "learning_rate": 1e-4,
            "dd_penalty": 0.1,
        },
    ]

    seed = 0  # 하이퍼파라미터 비교는 seed 0 하나만 사용

    valid_results = []
    test_results = []

    # =====================================================
    # 각 하이퍼파라미터 설정에 대해 Train -> Valid -> Test
    # =====================================================
    for cfg in configs:
        cfg_name = cfg["name"]
        lr = cfg["learning_rate"]
        dd_pen = cfg["dd_penalty"]

        print(f"\n=== Training PPO config: {cfg_name} "
              f"(lr={lr}, dd_penalty={dd_pen}) ===")

        # --- Train 환경 ---
        env_train = PortfolioEnvContinuous(
            price_df=train_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=dd_pen,
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env_train,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            seed=seed,
        )

        # 필요에 따라 timesteps 조절 (시간되면 150_000 정도, 급하면 80_000 정도)
        model.learn(total_timesteps=100_000)

        # --- 모델 저장 ---
        model_path = os.path.join(models_dir,
                                  f"ppo_cont_{cfg_name}_seed{seed}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # ---------- Validation 평가 ----------
        env_valid = PortfolioEnvContinuous(
            price_df=valid_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=dd_pen,
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        ret_valid = run_policy_on_env_returns(model, env_valid)
        stats_valid = performance_stats(ret_valid)
        stats_valid["config"] = cfg_name
        stats_valid["learning_rate"] = lr
        stats_valid["dd_penalty"] = dd_pen

        print("Validation stats:", stats_valid)
        valid_results.append(stats_valid)

        # ---------- Test 평가 ----------
        env_test = PortfolioEnvContinuous(
            price_df=test_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=dd_pen,
            turn_penalty=0.0005,
            max_weight=0.5,
            initial_nav=1.0,
        )

        ret_test = run_policy_on_env_returns(model, env_test)
        stats_test = performance_stats(ret_test)
        stats_test["config"] = cfg_name
        stats_test["learning_rate"] = lr
        stats_test["dd_penalty"] = dd_pen

        print("Test stats:", stats_test)
        test_results.append(stats_test)

    # ----- 결과 저장 -----
    df_valid = pd.DataFrame(valid_results).set_index("config")
    df_test  = pd.DataFrame(test_results).set_index("config")

    valid_log_path = os.path.join(logs_dir, "validation_results_ppo_hparam.csv")
    test_log_path  = os.path.join(logs_dir, "test_results_ppo_hparam.csv")

    df_valid.to_csv(valid_log_path)
    df_test.to_csv(test_log_path)

    print(f"\nValidation hparam results saved to {valid_log_path}")
    print(f"Test hparam results saved to {test_log_path}")


if __name__ == "__main__":
    main()
