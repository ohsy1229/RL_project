import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from env.portfolio_env_discrete import PortfolioEnvDiscrete
from utils.metrics import performance_stats
from utils.baselines import equal_weight_returns, buy_and_hold_returns


def run_policy_on_env(model, env: PortfolioEnvDiscrete) -> pd.Series:
    """
    학습된 RL 정책을 주어진 환경에서 한 번 전체 구간 실행하고,
    일별 '순수익률(net_ret)' 시계열을 반환하는 함수이다.
    (env.step()에서 거래비용까지 반영된 net_ret을 사용)
    """
    obs, info = env.reset()
    rets = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # env에서 net_ret, port_ret 둘 다 넘겨주므로,
        # 평가를 reward 정의와 맞추기 위해 net_ret 사용
        if "net_ret" in info:
            rets.append(info["net_ret"])

        if terminated or truncated:
            break

    return pd.Series(rets)


def main():
    # ---------- 경로 설정 ----------
    data_path = os.path.join("data", "processed", "prices.csv")
    results_dir = os.path.join("results")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ---------- 데이터 로드 ----------
    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    asset_cols = ["SEC", "HYU", "NAVER"]

    # Train / Valid / Test 분할
    train_df = price_df.loc[:"2016-12-30"]
    valid_df = price_df.loc["2017-01-02":"2019-12-30"]
    test_df = price_df.loc["2020-01-02":]

    print("Train length:", len(train_df),
          "Valid length:", len(valid_df),
          "Test length:", len(test_df))

    # 사용할 시드 목록
    seeds = [0, 42, 2024]
    results_valid = []

    # ==========================================================
    # 1) 각 seed별로 DQN 학습 (Train) + Validation 평가
    # ==========================================================
    for seed in seeds:
        print(f"\n=== [Phase 1] Training DQN with seed {seed} (Train -> Valid) ===")

        # --- Train 환경 ---
        env_train = PortfolioEnvDiscrete(
            price_df=train_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=0.5,
            trade_frac=0.1,
            max_weight=0.5,
            turn_penalty=0.001,
            initial_nav=1.0,
        )

        model = DQN(
            policy="MlpPolicy",
            env=env_train,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1_000,
            verbose=1,
            seed=seed,
        )

        model.learn(total_timesteps=80_000)

        # --- 모델 저장 ---
        model_path = os.path.join(models_dir, f"dqn_seed_{seed}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # --- Validation 환경에서 평가 (net_ret 기준) ---
        env_valid = PortfolioEnvDiscrete(
            price_df=valid_df,
            asset_cols=asset_cols,
            window_size=20,
            trading_cost=0.002,
            dd_penalty=0.5,
            trade_frac=0.1,
            max_weight=0.5,
            turn_penalty=0.001,
            initial_nav=1.0,
        )

        ret_valid = run_policy_on_env(model, env_valid)
        stats_valid = performance_stats(ret_valid)
        stats_valid["seed"] = seed

        print("Validation stats:", stats_valid)
        results_valid.append(stats_valid)

    # --- Validation 결과 정리 및 저장 ---
    df_valid = pd.DataFrame(results_valid).set_index("seed")
    print("\n[Phase 1] Validation performance (per seed):")
    print(df_valid)

    valid_log_path = os.path.join(logs_dir, "validation_results.csv")
    df_valid.to_csv(valid_log_path)
    print(f"\nValidation results saved to {valid_log_path}")

    # ==========================================================
    # 2) Sharpe 기준 best seed 선택 후, Train+Valid 전체로 재학습
    #    그리고 Test 구간 평가 + Baseline 비교
    # ==========================================================
    best_seed = df_valid["Sharpe"].idxmax()
    print(f"\n=== [Phase 2] Best seed (by Sharpe on Valid): {best_seed} ===")

    # Train+Valid 통합 데이터프레임
    train_full_df = pd.concat([train_df, valid_df])

    # --- Train_full 환경 ---
    env_train_full = PortfolioEnvDiscrete(
        price_df=train_full_df,
        asset_cols=asset_cols,
        window_size=20,
        trading_cost=0.002,
        dd_penalty=0.5,
        trade_frac=0.1,
        max_weight=0.5,
        turn_penalty=0.001,
        initial_nav=1.0,
    )

    model_best = DQN(
        policy="MlpPolicy",
        env=env_train_full,
        learning_rate=1e-4,
        buffer_size=80_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        verbose=1,
        seed=int(best_seed),
    )

    model_best.learn(total_timesteps=100_000)

    # --- 최종 모델 저장 ---
    best_model_path = os.path.join(models_dir, "dqn_best.zip")
    model_best.save(best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    # --- Test 환경에서 RL 평가 ---
    env_test = PortfolioEnvDiscrete(
        price_df=test_df,
        asset_cols=asset_cols,
        window_size=20,
        trading_cost=0.002,
        dd_penalty=0.5,
        trade_frac=0.1,
        max_weight=0.5,
        turn_penalty=0.001,
        initial_nav=1.0,
    )

    ret_test_rl = run_policy_on_env(model_best, env_test)
    rl_stats = performance_stats(ret_test_rl)
    rl_stats["Strategy"] = "RL_DQN"

    # --- Baseline: EW, BH (Test 구간, 동일 비용 구조) ---
    ew_ret = equal_weight_returns(
        test_df, asset_cols, trading_cost=0.002, rebalance_freq="M"
    )
    bh_ret = buy_and_hold_returns(test_df, asset_cols)

    ew_stats = performance_stats(ew_ret)
    ew_stats["Strategy"] = "EW"

    bh_stats = performance_stats(bh_ret)
    bh_stats["Strategy"] = "BH"

    # --- Test 결과 테이블 ---
    df_test = pd.DataFrame([rl_stats, ew_stats, bh_stats]).set_index("Strategy")
    print("\n[Phase 2] Test performance (RL vs Baselines):")
    print(df_test)

    test_log_path = os.path.join(logs_dir, "test_results.csv")
    df_test.to_csv(test_log_path)
    print(f"\nTest results saved to {test_log_path}")

    # --- Test 구간 RL NAV 저장 (net_ret 기준) ---
    nav_rl = (1 + ret_test_rl.fillna(0.0)).cumprod()
    nav_rl.name = "RL_DQN_NAV"
    nav_path = os.path.join(logs_dir, "test_nav_rl.csv")
    nav_rl.to_csv(nav_path)
    print(f"RL NAV (Test) saved to {nav_path}")

    # --- 참고: Train+Valid 구간 Baseline 성과도 한 번에 저장 (옵션) ---
    full_df = pd.concat([train_df, valid_df])
    ew_ret_full = equal_weight_returns(
        full_df, asset_cols, trading_cost=0.002, rebalance_freq="M"
    )
    bh_ret_full = buy_and_hold_returns(full_df, asset_cols)

    ew_stats_full = performance_stats(ew_ret_full)
    bh_stats_full = performance_stats(bh_ret_full)

    baseline_df = pd.DataFrame({"EW": ew_stats_full, "BH": bh_stats_full}).T
    baseline_log_path = os.path.join(logs_dir, "baseline_train_valid_results.csv")
    baseline_df.to_csv(baseline_log_path)
    print(f"\nBaseline (Train+Valid) results saved to {baseline_log_path}")


if __name__ == "__main__":
    main()
