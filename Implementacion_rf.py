"""
===============================================================================
Activar Enviroment:
source IA_AVANZADA/bin/activate (Linux)
.\IA_AVANZA\Scripts\activate (Windows)   
===============================================================================
Random Forest Regression para predecir producción de energía

Kevin Alejandro Ramírez Luna | A01711063@tec.mx
===============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import seaborn as sns

# Configuración para reproducibilidad
np.random.seed(42)

# 1. Carga y preparación de datos
print("Cargando el dataset...\n")

# Cambio del nombre de las columnas a las que habia en la documentación
nombres_columnas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure',
                    'Relative_humidity', 'Net_hourly_electrical_energy']

# Lectura del archivo .xlsx (elige una ruta)
df = pd.read_excel('C:/Users/kevin/Documents/ITESM/Inteligencia_Artifical_Avanzada/modulo_2_aprendizaje_maquina/Folds5x2_pp.xlsx', header=0, names=nombres_columnas)

# Limpieza de datos
df = df.dropna()
print(f"Dataset cargado: {df.shape[0]} instancias, {df.shape[1]} features")

# 2. Selección de variables
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
X = df[['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']].values
#X = df[['Temperature', 'Exhaust_vacuum']].values

y = df['Net_hourly_electrical_energy'].values

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 3. División de datos (Train + Validation + Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Validación: {X_val.shape[0]} muestras")
print(f" - Test: {X_test.shape[0]} muestras")

# 4. Hiperparámetros 
hyperparams = {
    'n_estimators': 200,           # Número de árboles
    'max_depth': 10,               # Profundidad máxima
    'min_samples_split': 5,        # Mínimo muestras para dividir
    'min_samples_leaf': 2,         # Mínimo de hojas (muestras)
    'max_features': 'sqrt',        # Características por división
    'random_state': 42,            # Reproducibilidad
    'n_jobs': -1                   # Usar todos los núcleos de la compu para el entrenamiento
}

print(f"\n=== Definición de parametros ===")
for key, value in hyperparams.items():
    print(f"  {key}: {value}")

# 5. Entrenamiento del modelo
print("\n=== Entrenamiento del modelo ===")
rf_model = RandomForestRegressor(**hyperparams)
rf_model.fit(X_train, y_train)

# 6. Evaluación del modelo
print("\n=== Evaluación del modelo ===")

# Predicciones en todos los sets
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Métricas de evaluación
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{set_name:12} - MSE: {mse:.6f}, R²: {r2:.4f}")

print_metrics(y_train, y_pred_train, "Entrenamiento")
print_metrics(y_val, y_pred_val, "Validación")
print_metrics(y_test, y_pred_test, "Test")

# 7. Importancia de características
print("\n=== Importancia de características ===")
feature_importance = rf_model.feature_importances_
#caracteristicas = ['Temperature', 'Exhaust_vacuum']
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
caracteristicas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']



for i, col in enumerate(caracteristicas):
    print(f"  {col}: {feature_importance[i]:.4f}")

# 8. Visualización de resultados
plt.figure(figsize=(15, 5))

# Train
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Train - R² = {r2_score(y_train, y_pred_train):.3f}')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.grid(True)

# Validacion
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_pred_val, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Validación - R² = {r2_score(y_val, y_pred_val):.3f}')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.grid(True)

# Test
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Test - R² = {r2_score(y_test, y_pred_test):.3f}')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.grid(True)

plt.tight_layout()
plt.show()


# 9. Predicción de ejemplo
print("\n=== Predicción de ejemplo ===")
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
ejemplo_X = np.array([[25.0, 60.0, 1013.0, 80.0]])  # Temperature, Exhaust_vacuum, Ambient_pressure, Relative_humidity
#ejemplo_X = np.array([[25.0, 60.0]])  # Temperature, Exhaust_vacuum
prediccion = rf_model.predict(ejemplo_X)

print("Inputs:")
for i, col in enumerate(caracteristicas):
    print(f"  {col}: {ejemplo_X[0,i]}")
print(f"Predicción: {prediccion[0]:.2f} MW")
