Forest Fire Weather Index (FWI) Prediction – Ridge Regression

1) Project Overview

This project predicts the Forest Fire Weather Index (FWI) using meteorological and environmental features. A Ridge Regression model is implemented to handle multicollinearity and tuned for optimal performance using hyperparameter alpha.

2) Project Features

2.1) Module 3: Feature Engineering & Scaling

  •	Selected input features correlated with the target FWI.
  •	Applied StandardScaler to normalize numerical features.
  •	Split dataset into input features (X) and target (y).
  •	Created training and test sets using an 80%-20% split.
  •	Saved the fitted scaler (scaler.pkl) for deployment consistency.

2.2) Module 4: Model Training – Ridge Regression

  •	Trained Ridge Regression to reduce variance and handle multicollinearity.
  •	Hyperparameter tuning performed on alpha values: [0.001, 0.01, 0.1, 1, 10, 100, 200].
  •	Performance metrics computed for each alpha on train and test data.
  •	Best alpha selected based on minimum Test MSE.
  •	Saved the best Ridge model as ridge.pkl.

2.3) Module 5: Evaluation & Optimization
Metrics computed:
  •	Mean Squared Error (MSE)
  • Root Mean Squared Error (RMSE)
  •	Mean Absolute Error (MAE)
  •	R² Score
Plots generated:
  • Predicted vs Actual FWI (Test set)
  •	Train MSE vs Alpha
  •	Train RMSE vs Alpha
  •	Train MAE vs Alpha
  •	Train R² vs Alpha
Overfitting/underfitting diagnosis:
  •	Underfitting → High train & test error
  •	Overfitting → Low train error & high test error
  •	Good Fit → Train & test errors similar

3) Hyperparameter Tuning – Alpha

Metrics tracked for multiple values of alpha:

  •	Train MSE, RMSE, MAE,R²
  •	Test MSE, RMSE, MAE, R²

Optimal alpha is selected based on:

✔ Low Train/Test MSE
✔ Low RMSE
✔ Low MAE
✔ High R²
✔ Balanced train vs test performance

4) Output

| File Name                     | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `Ridge_Model.py`              | Main Python script for training, evaluation, and saving the model |
| `ridge.pkl`                   | Saved Ridge Regression model (best alpha)                         |
| `scaler.pkl`                  | Saved StandardScaler used for feature normalization               |
| `RR(Actual vs Predicted).png` | Scatter plot showing Predicted vs Actual                          |
| `RR(Alpha vs MSE).png`        | Line plot showing Train MSE vs Alpha                              |
| `RR(Alpha vs RMSE).png`       | Line plot showing Train RMSE vs Alpha                             |
| `RR(Alpha vs MAE).png`        | Line plot showing Train MAE vs Alpha                              |
| `RR(Alpha vs R²).png`         | Line plot showing Train R² score vs Alpha                         |


4) Evaluation Metrics
Metric	Description:
MAE -->	Average absolute error
MSE -->	Penalizes large errors
RMSE -->Square root of MSE
R² Score -->Measures proportion of variance explained

	
5) Underfitting & Overfitting Check
  •	Underfitting: High train & test errors
  •	Overfitting: Low train error & high test error
  •	Good Fit: Train & test errors are similar
Best model is selected using the optimal alpha value.
