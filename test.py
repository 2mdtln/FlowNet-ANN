import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_excel('Data.xlsx')

X = data[['x', 'y']]
y_true = data[['U', 'V']]

model = load_model('out/final_model.keras')

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_true)

y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print("Test on full data:")
print(f"R²: {r2:.4f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min().min(), y_true.max().max()],
         [y_true.min().min(), y_true.max().max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs True Values')
plt.text(0.05, 0.95, f"R²: {r2:.4f}\nMSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}",
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
plt.grid(True)
plt.savefig('out/test_scatter.png')
plt.close()

y_pred_df = pd.DataFrame(y_pred, columns=['U_pred', 'V_pred'])
result_df = pd.concat([X.reset_index(drop=True), y_true.reset_index(drop=True), y_pred_df], axis=1)
result_df.to_excel('out/test_predictions.xlsx', index=False)

print("Predictions saved to test_predictions.xlsx and test_scatter.png")
