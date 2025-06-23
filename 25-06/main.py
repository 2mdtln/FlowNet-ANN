
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_excel("Data.xlsx")

# Giriş ve çıkışları ayır
X = df[['alfa', 'beta', 'x', 'y']].values
y = df[['U', 'V']].values

# Ölçeklendirme
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Veriyi ayır
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    verbose=1
)

# Kayıp grafiği (log)
plt.figure(figsize=(10, 6))
plt.semilogy(history.history['loss'], label='Eğitim Kaybı')
plt.semilogy(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title("Epoch'lara Göre MSE (Log Ölçek)")
plt.xlabel('Epoch')
plt.ylabel('MSE (Log)')
plt.legend()
plt.grid(True)
plt.savefig("loss_log_plot.png")
plt.close()

# Değerlendirme fonksiyonu
def evaluate(model, X, y_true, name):
    y_pred_scaled = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n📊 {name} Seti:")
    print(f"R²:   {r2:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    return y_true, y_pred

y_train_true, y_train_pred = evaluate(model, X_train, y_train, "Eğitim")
y_val_true, y_val_pred = evaluate(model, X_val, y_val, "Doğrulama")
y_test_true, y_test_pred = evaluate(model, X_test, y_test, "Test")

# Dağılım grafiği
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def plot_scatter(y_true, y_pred, title, filename):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    metrics_text = f"R²={r2:.3f}\nMSE={mse:.4f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}"

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    ax[0].plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'r--')
    ax[0].set_title(f'{title} - U Bileşeni')
    ax[0].set_xlabel('Gerçek U')
    ax[0].set_ylabel('Tahmin U')
    ax[0].grid()
    ax[0].text(0.05, 0.95, metrics_text, transform=ax[0].transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    ax[1].plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'r--')
    ax[1].set_title(f'{title} - V Bileşeni')
    ax[1].set_xlabel('Gerçek V')
    ax[1].set_ylabel('Tahmin V')
    ax[1].grid()
    ax[1].text(0.05, 0.95, metrics_text, transform=ax[1].transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(filename)
    plt.close()


plot_scatter(y_train_true, y_train_pred, "Eğitim Seti", "train_scatter.png")
plot_scatter(y_val_true, y_val_pred, "Doğrulama Seti", "val_scatter.png")
plot_scatter(y_test_true, y_test_pred, "Test Seti", "test_scatter.png")

model.save("model.keras")
