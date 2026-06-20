import pandas as pd

import numpy as np

import yfinance as yf

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

 

# -------------------- INPUT --------------------

user_input = input("Enter Stock (e.g., RELIANCE or TCS): ").upper()

 

# Auto add .NS if missing

if "." not in user_input:

    STOCK_NAME = user_input + ".NS"

else:

    STOCK_NAME = user_input

 

# -------------------- LOAD DATA --------------------

data = yf.download(STOCK_NAME, period="5y", interval="1d", auto_adjust=True)

 

if data.empty:

    raise ValueError(f"Invalid stock name: {STOCK_NAME}")

 

data.reset_index(inplace=True)

 

# -------------------- FEATURE ENGINEERING --------------------

data['Open-Close'] = data['Open'] - data['Close']

data['High-Low'] = data['High'] - data['Low']

 

data['MA10'] = data['Close'].rolling(10).mean()

data['MA50'] = data['Close'].rolling(50).mean()

data['Return'] = data['Close'].pct_change()

 

data.dropna(inplace=True)

 

# -------------------- TARGET --------------------

data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

data = data[:-1]

 

# -------------------- FEATURES --------------------

X = data[['Open-Close', 'High-Low', 'MA10', 'MA50', 'Return']]

y = data['Target']

 

# -------------------- SCALING --------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

 

# -------------------- TRAIN --------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled, y, test_size=0.25, random_state=44

)

 

params = {'n_neighbors': np.arange(3, 25)}

 

model = GridSearchCV(KNeighborsClassifier(), params, cv=5)

model.fit(X_train, y_train)

 

print("\nModel Accuracy:", model.score(X_test, y_test))

 

# -------------------- FINAL PREDICTION --------------------

latest_data = data.iloc[-1]

 

open_close = latest_data['Open-Close'].item()

high_low = latest_data['High-Low'].item()

ma10 = latest_data['MA10'].item()

ma50 = latest_data['MA50'].item()

ret = latest_data['Return'].item()

 

latest_features = np.array([open_close, high_low, ma10, ma50, ret]).reshape(1, -1)

latest_features = scaler.transform(latest_features)

 

prediction = model.predict(latest_features)[0]

 

# -------------------- DECISION --------------------

if prediction == 1:

    decision = "BUY"

    reason = "Uptrend detected using moving averages and returns."

elif prediction == -1:

    decision = "SELL"

    reason = "Downtrend detected using moving averages and returns."

else:

    decision = "HOLD"

    reason = "No clear signal."

 

if abs(ret) < 0.001:

    decision = "HOLD"

    reason = "Very low return → market stable."

 

# -------------------- LIVE PRICE --------------------

ticker = yf.Ticker(STOCK_NAME)

live_price = ticker.history(period="1d")['Close'].iloc[-1]

 

# -------------------- OUTPUT --------------------

print("\n========== FINAL OUTPUT ==========")

print("Stock:", STOCK_NAME)

print("Live Market Price:", live_price)

print("Final Decision:", decision)

print("Reason:", reason)

print("=================================")
