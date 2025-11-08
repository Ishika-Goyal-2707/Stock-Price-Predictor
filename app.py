from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper().strip()

    try:
        # ✅ Fetch up-to-date data till today
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(ticker, start='2023-01-01', end=end_date, progress=False)

        if data.empty:
            return jsonify({'error': f'No data found for {ticker}. Please check the symbol.'})

        # ✅ Flatten multi-index columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        df = data.copy()

        # ✅ Add features
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # ✅ Prepare data for model
        df['Day'] = np.arange(len(df))
        X = df[['Day']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)

        # ✅ Predict next 5 days
        future_days = np.arange(len(df), len(df) + 5).reshape(-1, 1)
        future_preds = model.predict(future_days)
        future_dates = [(df.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]

        # ✅ Clean for JSON
        df = df.replace([float('inf'), -float('inf')], None).where(pd.notnull(df), None)

        chart_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'close': [float(x) if x is not None else 0 for x in df['Close'].values.tolist()],
            'ma10': [float(x) if x is not None else 0 for x in df['MA10'].values.tolist()],
            'ma30': [float(x) if x is not None else 0 for x in df['MA30'].values.tolist()],
            'pred_dates': future_dates,
            'pred_values': [float(x) for x in future_preds.tolist()]
        }

        return jsonify({'chart_data': chart_data})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
