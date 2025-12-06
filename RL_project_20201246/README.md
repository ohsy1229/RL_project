# 강화학습 기반 3자산 포트폴리오 트레이딩  
### DQN(이산 행동) & PPO(연속 행동)를 활용한 동적 자산배분 전략 구현

이 프로젝트는 한국 주식 3종목(삼성전자·현대차·NAVER)을 대상으로  
**강화학습(Reinforcement Learning, RL)** 기반의 자산배분 전략을 구축하고,  
전통적 전략인 **Equal Weight(EW)** 및 **Buy & Hold(BH)** 와 비교하여  
RL이 초과성과를 낼 수 있는지 평가한다.

본 연구에서는  
- **DQN(이산형 행동 기반)**  
- **PPO(연속형 행동 기반)**  

두 가지 RL 알고리즘을 모두 적용하였으며,  
각 환경(Environment)에 맞는 state/action/reward를 설계하여 실험을 진행하였다.

---

## 프로젝트 구조

RL-Portfolio-Allocation/
│
├── data/
│ ├── raw/
│ └── processed/ ← prices.csv
│
├── notebooks/
| ├── explotarion.ipynb
│ └── rl_project_analysis.ipynb ← DQN + PPO 통합 분석 노트북
│
├── src/
│ ├── env/
│ │ ├── portfolio_env.py ← DQN용 이산형 환경
│ │ └── portfolio_env_continuous.py ← PPO용 연속형 환경
│ │
│ ├── utils/
│ │ ├── baselines.py ← EW/BH baseline 구현 (비용 반영)
│ │ ├── metrics.py ← 성과지표 계산 함수
│ │
│ ├── train_dqn_discrete.py ← seed 학습 + best seed 선택 + test 평가
│ ├── train_ppo_continuous.py  ← seed 학습 + best seed 선택 + test 평가
│ ├── analyze_results_dqn.py
│ ├── analyze_results_ppo.py
│
├── results/ ← DQN 결과 저장
│ ├── models/
│ ├── logs/
│ └── figures/
│
├── results_continuous/ ← PPO 결과 저장
│ ├── models/
│ ├── logs/
│ └── figures/
│
└── README.md

---

## 데이터 구성

### 사용 종목

| 티커 | 종목 |
|------|------|
| SEC  | 삼성전자 |
| HYU  | 현대자동차 |
| NAVER | NAVER |

### 기간 분할

- **Train**: ~2005–2016  
- **Validation**: 2017–2019  
- **Test**: 2020–2023  

모든 트레이딩은 **일 단위**로 수행된다.

---

## 환경(Environment) 설계

### 1) **State (상태)**

#### DQN (Discrete 환경)
- 최근 20일 × 3자산 수익률  
- 현재 포트폴리오 비중  

#### PPO (Continuous 환경, 확장된 State)
- 최근 20일 수익률  
- 현재 weight  
- 20일/60일 모멘텀  
- 20일 변동성  
- (옵션) 시장 모멘텀 / 변동성  

---

### 2) **Action (행동)**

#### DQN 이산 행동
0: Hold
1: Buy SEC
2: Sell SEC
3: Buy HYU
4: Sell HYU
5: Buy NAVER
6: Sell NAVER

#### PPO 연속 행동
a ∈ [-1, 1]^3 → weight 신호

→ [0, max_weight] 범위로 매핑 후 → 합이 1이 되도록 정규화

---

### 3) 보상(Reward) 설계

강화학습 성능은 보상 함수 설계에 큰 영향을 받으므로  
DQN(이산 행동)과 PPO(연속 행동)에 서로 다른 보상 구조를 사용했다.

---

#### **DQN 환경 (Discrete Action Environment)**  
DQN 환경은 **선형 순수익률(net return)** 기반 보상을 사용한다.

reward = net_ret - dd_penalty * drawdown - turn_penalty * turnover

- **net_ret** = 포트폴리오 수익률 − 거래비용  
- Drawdown 및 Turnover 패널티는 위험 과도 노출을 억제하기 위한 규제항  
- NAV 업데이트 방식  
NAV_t = NAV_{t-1} * (1 + net_ret)


이 구조는 값이 안정적이고 DQN 학습에 적합하며,
기본적인 "순이익 최대화" 형태의 보상 설계를 따른다.

---

#### **PPO 환경 (Continuous Action Environment)**  
PPO는 연속 행동(action vector) 기반이므로  
보상 분포를 안정화하기 위해 **로그 수익률(log-return)** 기반 보상을 사용했다.

reward = log(1 + net_ret) - dd_penalty * drawdown - turn_penalty * turnover

- 로그 수익률은 극단값(outlier)에 덜 민감  
- PPO와 같은 policy gradient 계열 알고리즘에서 학습 안정성이 향상됨  
- net_ret이 -1 이하로 가는 비정상 상황을 막기 위해 내부적으로  
  `net_ret = max(net_ret, -0.99)` 형태의 안전 처리 적용 가능

---

## Baseline 비교 전략

### **Equal Weight (EW)**  
- 매월 리밸런싱  
- RL과 동일한 거래비용 적용 → “공정한 비교”  

### **Buy & Hold (BH)**  
- 초기 매수 후 리밸런싱 없음  
- 거래비용 없음  

---

## 실험 절차

### 1) Train Phase
- Seeds: **0, 42, 2024**
- 각 seed로 학습 후 validation 성과 평가 (Sharpe 기준)

### 2) Select Best Seed
- Validation Sharpe가 가장 높은 seed 선택

### 3) Retrain
- Train + Validation 전체 기간으로 다시 학습

### 4) Test Evaluation
- 최종 모델로 Test 구간 NAV / 성과지표 평가  
- Baseline(EW, BH)과 비교

---

# 주요 실험 결과

## **DQN (Discrete)**

| 전략 | CAGR | Vol | Sharpe | MDD |
|------|--------|--------|----------|---------|
| RL_DQN | 0.0838 | 0.223 | 0.472 | -0.416 |
| EW     | 0.0900 | 0.206 | 0.521 | -0.433 |
| BH     | 0.0739 | 0.208 | 0.445 | -0.447 |

### 해석
- RL은 상승장 흐름을 일정 부분 따라가지만  
- EW 대비 **반응이 느리고**, 거래비용이 누적되면서 성과가 제한됨  
- Drawdown 방어도 baseline 수준에는 미치지 못함

---

## **PPO (Continuous)**

| 전략 | CAGR | Vol | Sharpe | MDD |
|------|--------|--------|----------|---------|
| RL_PPO_CONT | 0.0593 | 0.208 | 0.380 | -0.439 |
| EW          | 0.0900 | 0.206 | 0.521 | -0.434 |
| BH          | 0.0739 | 0.208 | 0.445 | -0.447 |

### 해석
- 연속 행동(Continuous action)은 자유도가 매우 높아 PPO가 안정적 정책을 학습하기 어려움  
- 강화된 state로 일부 개선되었으나 baseline 초과 성과는 미달  
- 금융 시계열 환경에서는 **Action 설계·Feature 설계·Reward 안정성**이 특히 중요함  

---

## 📉 핵심 요약

1. RL(DQN/PPO)은 **기초적인 추세 추종 능력**은 학습했지만  
   baseline(EW)을 안정적으로 넘어서지는 못했다.
2. Continuous PPO는 자유도가 너무 큰 탓에 **정책 안정성 문제** 발생.
3. State와 Reward 설계가 성능에 결정적 영향을 미친다.
4. 금융 환경은 RL 적용 난이도가 높은 영역이며, **환경 설계가 곧 성능**이다.

---

## 실행 방법

### DQN 학습 + 테스트
python src/train_dqn_all.py
python src/analyze_results_dqn.py


### PPO 연속액션 학습 + 테스트
python src/train_ppo_continuous.py
python src/analyze_results_ppo.py


### 분석 노트북
notebooks/rl_project_analysis.ipynb

(DQN + PPO 결과, 그래프 비교, 행동 패턴 시각화 포함)

---