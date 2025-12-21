from flask import Flask, render_template, request, redirect, url_for
import psycopg2
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import json
import yfinance as yf
from stable_baselines3 import PPO
import gymnasium as gym
from tensorflow.keras.models import load_model
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

# ==================================================
# BASIC SETUP
# ==================================================
load_dotenv()
app = Flask(__name__)

LOOKBACK = 60
LEVERAGE = 5
CACHE_FILE = "df_cache.pkl"
CACHE_MAX_AGE = 600  # 10 Minutes cache for heavy indicators

# PPO training columns (LOCKED)
OBS_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "ma10", "ma50",
    "ema12", "ema26",
    "macd", "macd_signal",
    "rsi",
    "bb_middle", "bb_upper", "bb_lower",
    "volume_sma20",
    "lstm_pred_return"
]

OBS_SIZE = LOOKBACK * len(OBS_COLUMNS) + 4

# ==================================================
# DB CONNECTION
# ==================================================
def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        port=os.getenv("DB_PORT")
    )

# ==================================================
# LOAD MODELS (ONCE)
# ==================================================
ppo_model = PPO.load("btc_hybrid_final_300k")
lstm_model = load_model("btc_lstm_predictor.keras")

with open("btc_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==================================================
# LIVE PRICE HELPER (FRESH FETCH)
# ==================================================
def get_live_btc_price():
    """Fetches only the latest price without processing indicators."""
    try:
        # ticker.fast_info['last_price'] is the fastest way
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return round(float(data['Close'].iloc[-1]), 2)
        return 0.0
    except Exception as e:
        print(f"Live Price Error: {e}")
        return 0.0

# ==================================================
# TECHNICAL INDICATORS
# ==================================================
def add_indicators(df):
    df = df.copy()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_middle"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * std
    df["bb_lower"] = df["bb_middle"] - 2 * std
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    return df

# ==================================================
# LSTM PREDICTION
# ==================================================
LSTM_FEATURES = [
    "close", "volume", "ma10", "ma50",
    "ema12", "ema26", "macd", "macd_signal",
    "rsi", "bb_middle", "bb_upper", "bb_lower",
    "volume_sma20"
]

def lstm_pred_return(idx, df):
    if idx < LOOKBACK:
        return 0.0
    try:
        seq = df[LSTM_FEATURES].iloc[idx-LOOKBACK:idx].values
        seq_scaled = scaler.transform(seq).reshape(1, LOOKBACK, len(LSTM_FEATURES))
        pred_scaled = lstm_model.predict(seq_scaled, verbose=0)[0][0]
        dummy = np.zeros((1, len(LSTM_FEATURES)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        cur = df.iloc[idx]["close"]
        return (pred_price - cur) / cur if cur > 0 else 0.0
    except:
        return 0.0

# ==================================================
# LOAD AND PROCESS DATA (CACHED)
# ==================================================
def load_and_process_data():
    if os.path.exists(CACHE_FILE):
        cache_time = os.path.getmtime(CACHE_FILE)
        if time.time() - cache_time < CACHE_MAX_AGE:
            return pd.read_pickle(CACHE_FILE)

    print("Refreshing heavy data cache...")
    df = yf.download("BTC-USD", period="7d", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = df.columns.str.lower()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_indicators(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["lstm_pred_return"] = [lstm_pred_return(i, df) for i in range(len(df))]
    df = df[OBS_COLUMNS].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(CACHE_FILE)
    return df

# ==================================================
# RL ENVIRONMENT
# ==================================================
class HybridBitcoinFuturesEnv(gym.Env):
    def __init__(self, df, initial_balance, max_loss_pct):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_loss_pct = max_loss_pct / 100
        self.min_balance = initial_balance * (1 - self.max_loss_pct)
        self.leverage = LEVERAGE
        self.lookback = LOOKBACK
        self.current_step = None
        self.position = 0
        self.entry_price = 0.0
        self.trade_capital_pct = 0.1
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.current_step = np.random.randint(self.lookback, len(self.df) - 2)
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.df.iloc[self.current_step - self.lookback : self.current_step].values.flatten()
        extra = np.array([self.balance / self.initial_balance, float(self.position), 0.0, self.df.iloc[self.current_step]["lstm_pred_return"]])
        obs = np.concatenate([window, extra])
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0
        trade_capital = self.balance * self.trade_capital_pct
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price

        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price
            if self.position == -1: pnl_pct = -pnl_pct
            self.balance += pnl_pct * trade_capital * self.leverage

        done = self.balance <= self.min_balance
        self.current_step += 1
        if self.current_step >= len(self.df) - 1: done = True
        return self._get_obs(), reward, done, False, {}

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def dashboard():
    df = load_and_process_data()
    live_price = get_live_btc_price()  # Humesha fresh price
    
    env = HybridBitcoinFuturesEnv(df, 100000, 10)
    obs, _ = env.reset()
    action, _ = ppo_model.predict(obs, deterministic=True)
    signal = {0: "HOLD", 1: "BUY", 2: "SELL"}[int(action)]
    
    return render_template("index.html", price=live_price, signal=signal)

@app.route("/run-backtest", methods=["POST"])
def run_backtest():
    initial_inr = float(request.form["initial_inr"])
    max_loss = float(request.form["max_loss"])
    df = load_and_process_data()
    env = HybridBitcoinFuturesEnv(df, initial_balance=initial_inr, max_loss_pct=max_loss)
    obs, _ = env.reset()
    done = False
    actions = []
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        actions.append(int(action))
    
    final_balance = env.balance
    total_return = ((final_balance - initial_inr) / initial_inr) * 100
    unique, counts = np.unique(actions, return_counts=True) if actions else ([], [])
    action_map = {0: 'Hold', 1: 'Long', 2: 'Short'}
    total = len(actions) if actions else 1
    dist = {action_map.get(a, 'Unknown'): (c / total * 100) for a, c in zip(unique, counts)}

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO backtest_results (initial_balance, final_balance, total_return, max_drawdown, actions_json, notes) VALUES (%s, %s, %s, %s, %s, %s)", 
                (initial_inr, final_balance, total_return, max_loss, json.dumps(dist), "Hybrid PPO + LSTM"))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("backtests"))

@app.route("/backtests")
def backtests():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, initial_balance, final_balance, total_return, max_drawdown, notes, actions_json
        FROM backtest_results ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        timestamp, initial, final, ret, max_dd, notes, actions_json = row

        # Super safe JSON parsing
        dist = {"Hold": 0.0, "Long": 0.0, "Short": 0.0}  # Default safe dict

        if actions_json:
            try:
                parsed = json.loads(actions_json)
                if isinstance(parsed, dict):
                    # Update only valid keys
                    dist["Hold"] = float(parsed.get("Hold", 0.0))
                    dist["Long"] = float(parsed.get("Long", 0.0))
                    dist["Short"] = float(parsed.get("Short", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # Agar kuch galat hai to default hi rahega

        results.append({
            "timestamp": timestamp,
            "initial_balance": initial,
            "final_balance": final,
            "total_return": ret,
            "max_drawdown": max_dd,
            "notes": notes,
            "dist": dist
        })

    return render_template("backtest.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)