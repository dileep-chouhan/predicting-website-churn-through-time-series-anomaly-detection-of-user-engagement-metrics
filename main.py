import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
dates = pd.date_range(start='2023-01-01', periods=365)
daily_users = 1000 + 200 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 50, 365)
daily_sessions = 2 * daily_users + np.random.normal(0, 100, 365)
daily_engagement = daily_sessions / daily_users
#Introduce a simulated churn event
daily_users[200:250] -= 200
daily_users = np.maximum(0, daily_users) #Ensure no negative user counts
df = pd.DataFrame({'Date': dates, 'DailyUsers': daily_users, 'DailySessions': daily_sessions, 'DailyEngagement': daily_engagement})
# --- 2. Data Cleaning and Preprocessing (minimal in this synthetic example) ---
#Check for missing values (unlikely in synthetic data, but good practice)
if df.isnull().values.any():
    print("Warning: Missing values detected. Consider imputation.")
    #Example imputation (replace with more sophisticated methods if needed):
    #df.fillna(method='ffill', inplace=True)
# --- 3. Time Series Analysis and Anomaly Detection ---
#Simple rolling average to smooth the data and highlight trends
window_size = 7
df['RollingAvgUsers'] = df['DailyUsers'].rolling(window=window_size, center=True).mean()
df['RollingAvgEngagement'] = df['DailyEngagement'].rolling(window=window_size, center=True).mean()
#Anomaly detection using standard deviation (a simple method, replace with more advanced methods if needed)
threshold = 2 # Adjust this threshold based on your needs
df['UserAnomaly'] = (df['DailyUsers'] < df['RollingAvgUsers'] - threshold * df['DailyUsers'].std())
df['EngagementAnomaly'] = (df['DailyEngagement'] < df['RollingAvgEngagement'] - threshold * df['DailyEngagement'].std())
# --- 4. Visualization ---
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['DailyUsers'], label='Daily Users')
plt.plot(df['Date'], df['RollingAvgUsers'], label='Rolling Average Users')
plt.scatter(df[df['UserAnomaly']]['Date'], df[df['UserAnomaly']]['DailyUsers'], color='red', label='Anomalies')
plt.title('Daily Users with Rolling Average and Anomalies')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(df['Date'], df['DailyEngagement'], label='Daily Engagement')
plt.plot(df['Date'], df['RollingAvgEngagement'], label='Rolling Average Engagement')
plt.scatter(df[df['EngagementAnomaly']]['Date'], df[df['EngagementAnomaly']]['DailyEngagement'], color='red', label='Anomalies')
plt.title('Daily Engagement with Rolling Average and Anomalies')
plt.legend()
plt.tight_layout()
plt.savefig('anomaly_detection_plot.png')
print("Plot saved to anomaly_detection_plot.png")
# --- 5.  Predictive Modeling (Illustrative -  More advanced models are needed for robust prediction)---
#  This section illustrates a simple approach.  For real-world applications, explore ARIMA, Prophet, LSTM, etc.
#  We'll use a simple linear regression on the rolling average to illustrate prediction (highly simplified)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Prepare data for prediction (using rolling average to smooth the data)
X = np.arange(len(df['RollingAvgUsers'])).reshape(-1, 1)
y = df['RollingAvgUsers'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #No shuffle for time series
model = LinearRegression()
model.fit(X_train, y_train)
future_x = np.arange(len(df['RollingAvgUsers']), len(df['RollingAvgUsers']) + 30).reshape(-1, 1) #predict next 30 days
predicted_users = model.predict(future_x)
#Visualization of prediction (highly simplified)
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['RollingAvgUsers'], label='Actual')
plt.plot(pd.date_range(start=df['Date'].max()+pd.Timedelta(days=1), periods=30), predicted_users, label='Predicted')
plt.title('Simple Prediction of Daily Users')
plt.legend()
plt.savefig('simple_prediction.png')
print("Plot saved to simple_prediction.png")