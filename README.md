# 🤖 Hybrid Crypto Algo Trader (LSTM + Reinforcement Learning)

A **full-fledged algorithmic trading system** for **Bitcoin (BTC)** that combines:

- 📈 **LSTM price prediction**
- 🧠 **Reinforcement Learning (PPO)**
- ⚡ **Leverage-based futures trading**
- 🧪 **Backtesting on 2 years of real BTC data**

This project demonstrates how **prediction models + decision-making agents**
can be combined to build a realistic **AI-powered trading bot**.

> ⚠️ DISCLAIMER  
> This project is strictly for **educational & research purposes only**.  
> It is **NOT financial advice**.  
> Do NOT use this bot for real-money trading.  
> Crypto trading involves extreme risk.

---

## 🧠 Core Idea

Most trading bots do **either**:
- Price prediction (ML / DL) ❌
- Buy/Sell automation (RL) ❌

This project does **BOTH** ✅

### 🔥 Hybrid Architecture
1. **LSTM model**
   - Predicts next-period BTC price movement
   - Generates expected return (`lstm_pred_return`)
2. **Reinforcement Learning agent (PPO)**
   - Uses:
     - Market state
     - Account state
     - LSTM predicted return
   - Decides:
     - Long / Short / Hold
     - Position switching
     - Risk-aware execution

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras (LSTM)**
- **Reinforcement Learning**
- **Stable-Baselines3 (PPO)**
- **Gymnasium (Custom Environment)**
- **CCXT**
- **yFinance**
- **NumPy / Pandas / Matplotlib**

---

## 📊 Trading Environment (Key Highlight)

### Action Space
| Action | Meaning |
|------|--------|
| 0 | Hold |
| 1 | Long |
| 2 | Short |

### Observation Space
- Last **60 candles**
- OHLCV + Technical Indicators
- Account balance ratio
- Current position
- Unrealized PnL
- **LSTM predicted return**

> This makes the agent **prediction-aware**, not blind.

---

## ⚙️ Risk & Execution Logic

- Futures trading with **5x leverage**
- Transaction fees included
- Unrealized PnL shaping reward
- Automatic position closing
- Capital protection logic

---

## 📈 Backtesting Result (2 Years – Daily BTC Data)

```text
Initial Balance: $10,000
Final Balance:   ~$14,400 (training phase)
Return:          +44%

Extended Backtest:
Final Balance:   ~$1,437,308
Total Return:    +14,273%
