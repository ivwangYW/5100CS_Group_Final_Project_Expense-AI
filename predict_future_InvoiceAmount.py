import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin  # 导入 BaseEstimator 和 RegressorMixin
import matplotlib.pyplot as plt
from datetime import datetime


class SeasonalTrendRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, period=365.25, frequency=1):
        self.period = period
        self.frequency = frequency
        self.trend_model = LinearRegression()
        self.seasonal_model = LinearRegression()

    def fit(self, X, y):
        # Fit the trend model
        self.trend_model.fit(X, y)

        # Detrend the data
        trend = self.trend_model.predict(X)
        detrended_y = y - trend

        # Fit the seasonal model
        X_seasonal = self._prepare_seasonal_features(X)
        self.seasonal_model.fit(X_seasonal, detrended_y)

        return self

    def predict(self, X):
        # Predict the trend
        trend = self.trend_model.predict(X)

        # Predict the seasonality
        X_seasonal = self._prepare_seasonal_features(X)
        seasonality = self.seasonal_model.predict(X_seasonal)

        # Combine the trend and seasonality
        return trend + seasonality

    def _prepare_seasonal_features(self, X):
        # Create seasonal features based on sine and cosine functions
        X_seasonal = np.column_stack([
            np.sin(2 * np.pi * self.frequency * X.squeeze() / self.period),
            np.cos(2 * np.pi * self.frequency * X.squeeze() / self.period)
        ])
        return X_seasonal


def preprocess_data(df, employeeID, expenseCategory):
    df_filtered = df[(df['EmployeeID'] == employeeID) & (df['ExpenseCategory'] == expenseCategory)].copy()
    df_filtered['InvoiceDate'] = pd.to_datetime(df_filtered['InvoiceDate'])
    df_filtered['DayOfYear'] = df_filtered['InvoiceDate'].dt.dayofyear
    df_filtered['Month'] = df_filtered['InvoiceDate'].dt.month
    df_filtered['DayOfWeek'] = df_filtered['InvoiceDate'].dt.dayofweek
    df_filtered['IsMonthEnd'] = df_filtered['InvoiceDate'].dt.is_month_end.astype(int)
    return df_filtered


def predictInvoiceAmount(futureDate, employeeID, expenseCategory, df_reimbursementHistory):
    df = preprocess_data(df_reimbursementHistory, employeeID, expenseCategory)
    df['InvoiceDateInt'] = (df['InvoiceDate'] - df['InvoiceDate'].min()).dt.days
    X_trend = df[['InvoiceDateInt']].values
    y = df['InvoiceAmount']

    # Create and fit the seasonal trend model
    seasonal_trend_model = SeasonalTrendRegressor(period=365.25/4, frequency=1)
    seasonal_trend_model.fit(X_trend, y)

    # Predict the trend and seasonal components
    trend_pred = seasonal_trend_model.predict(X_trend)

    # Generate future dates and predict
    last_historical_date = df['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_historical_date, end=futureDate, freq='D')
    future_df = pd.DataFrame({
        'InvoiceDate': future_dates,
        'InvoiceDateInt': (future_dates - df['InvoiceDate'].min()).days
    })
    X_future_trend = future_df[['InvoiceDateInt']].values
    future_trend_pred = seasonal_trend_model.predict(X_future_trend)

    # Plot the historical and future data
    plt.figure(figsize=(12, 6))
    plt.scatter(df['InvoiceDate'], y, color='blue', label='Historical Invoice Amounts')
    plt.plot(df['InvoiceDate'], trend_pred, color='green', label='Trend + Seasonality')
    plt.plot(future_df['InvoiceDate'], future_trend_pred, color='red', linestyle='--',
             label='Future Trend + Seasonality Prediction')
    plt.xlabel('Date')
    plt.ylabel('Invoice Amount')
    plt.title(f'Invoice Amount Forecast for Employee {employeeID}, Category {expenseCategory}')
    plt.legend()
    plt.show()

    return future_trend_pred[-1]

if __name__ == "__main__":
    test_cases = [
        {"employeeID": "E000002", "expenseCategory": "Phone & Internet"},
        {"employeeID": "E000435", "expenseCategory": "Phone & Internet"},
        {"employeeID": "P000009", "expenseCategory": "Travel"},
        {"employeeID": "P000002", "expenseCategory": "Travel"},
    ]

    future_dates = ["2025-01-01", "2030-01-01"]

    # 加载数据
    df_reimbursementHistory = pd.read_csv('database_data/Table_ReimbursementRequestRecords.csv')

    # 测试函数
    def run_tests():
        for case in test_cases:
            employeeID = case["employeeID"]
            expenseCategory = case["expenseCategory"]
            for futureDate in future_dates:
                predicted_amount = predictInvoiceAmount(futureDate, employeeID, expenseCategory, df_reimbursementHistory)
                print(f"Test Case - EmployeeID: {employeeID}, ExpenseCategory: {expenseCategory}, FutureDate: {futureDate}")
                print(f"Predicted Invoice Amount: {predicted_amount}\n")

    # 运行测试
    run_tests()

