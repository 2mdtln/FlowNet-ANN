import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

def load_and_preprocess_data(filename):
    data = pd.read_excel(filename)
    X = data[['x', 'y']].values
    y = data[['U', 'V']].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def build_model(input_dim=2):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(32),
        LeakyReLU(alpha=0.2),
        Dense(2)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_loss(history, filename='out/final_loss_plot.png'):
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    X_scaled, y_scaled, _, scaler_y = load_and_preprocess_data('Data.xlsx')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    
    model = build_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        verbose=2
    )
    
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    data = pd.read_excel('Data.xlsx')
    y_pred_df = pd.DataFrame(y_pred, columns=['U_pred', 'V_pred'])
    result_df = pd.concat([data[['x', 'y']], data[['U', 'V']], y_pred_df], axis=1)
    result_df.to_excel('out/final_prediction.xlsx', index=False)
    
    model.save('out/final_model.keras')    
    plot_loss(history)

if __name__ == '__main__':
    main()
