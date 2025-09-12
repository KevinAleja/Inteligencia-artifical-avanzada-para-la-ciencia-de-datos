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
from sklearn.model_selection import GridSearchCV
# ======================
# Librerias para graficar
# ======================
#from sklearn import tree
#import graphviz 


# Configuración para reproducibilidad
np.random.seed(42)

# 1. Carga y preparación de datos
print("Cargando el dataset...\n")

# Cambio del nombre de las columnas a las que habia en la documentación
nombres_columnas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure',
                    'Relative_humidity', 'Net_hourly_electrical_energy']

# Lectura del archivo .xlsx
#df = pd.read_excel('/home/kevinalejarmzl/Documentos/Inteligencia_Artifical_Avanzada/modulo_2_aprendizaje_maquina/Folds5x2_pp.xlsx', header=0, names=nombres_columnas)
df = pd.read_excel('C:/Users/kevin/Documents/ITESM/Inteligencia_Artifical_Avanzada/modulo_2_aprendizaje_maquina/Folds5x2_pp.xlsx', header=0, names=nombres_columnas)

# Limpieza de datos
df = df.dropna() # Eliminamos algún valor nulo si es que lo hubiera 
print(f"Dataset cargado: {df.shape[0]} instancias, {df.shape[1]} features")

# 2. Selección de variables (misma que en el modelo lineal)
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
X = df[['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']].values
#X = df[['Temperature', 'Exhaust_vacuum']].values

y = df['Net_hourly_electrical_energy'].values

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 3. División de datos para entrenamiento, validación y prueba 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)

print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Prueba: {X_test.shape[0]} muestras")
print(f" - Test: {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")

# 4. Entrenamiento del modelo Random Forest
print("\n=== Entrenamiento del modelo Random Forest ===")

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}


# Crear el modelo base
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Búsqueda de hiperparámetros con CV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # Numero de divisiones 
    scoring='r2', 
    n_jobs=-1,  # Usar todos los cores disponibles del equipo
    verbose=1
)

# Entrenar el modelo con búsqueda de hiperparámetros
print("Realizando búsqueda de hiperparámetros...")
grid_search.fit(X_train, y_train)

# Mejores hiperparámetros encontrados
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor score de validación: {grid_search.best_score_:.4f}")

# Mejor modelo
best_rf = grid_search.best_estimator_

# 5. Evaluación del modelo
print("\n=== Evaluación del modelo Random Forest ===")

y_pred_train_rf = best_rf.predict(X_train)
y_pred_val_rf = best_rf.predict(X_val)
y_pred_test_rf = best_rf.predict(X_test)

# Métricas de evaluación
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{set_name:12} - MSE: {mse:.6f}, R²: {r2:.4f}")
    return r2

r2_train_rf = print_metrics(y_train, y_pred_train_rf, "Entrenamiento")
r2_val_rf = print_metrics(y_val, y_pred_val_rf, "Validación")          
r2_test_rf = print_metrics(y_test, y_pred_test_rf, "Test")             

# 6. Importancia de características
print("\n=== Importancia de características ===")
feature_importance = best_rf.feature_importances_

# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
caracteristicas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']
#caracteristicas = ['Temperature', 'Exhaust_vacuum']

for i, col in enumerate(caracteristicas):
    print(f"  {col}: {feature_importance[i]:.4f}")

# 7. Visualización de resultados
plt.figure(figsize=(15, 5))

# Predicciones vs Valores reales (train)
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_pred_train_rf, alpha=0.5, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title(f'Train RF: R² = {r2_train_rf:.3f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones (MW)')
plt.grid(True)

# Predicciones vs Valores reales (validación)
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_pred_val_rf, alpha=0.5, color='green')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title(f'Validación RF: R² = {r2_val_rf:.3f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones (MW)')
plt.grid(True)

# Predicciones vs Valores reales (test)
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test_rf, alpha=0.5, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Test RF: R² = {r2_test_rf:.3f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones (MW)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Predicción de ejemplo
print("\n=== PREDICCIÓN DE EJEMPLO ===")
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
ejemplo_X = np.array([[25.0, 60.0, 1013.0, 80.0]])  # Temperature, Exhaust_vacuum, Ambient_pressure, Relative_humidity
#ejemplo_X = np.array([[25.0, 60.0]])

prediccion_rf = best_rf.predict(ejemplo_X)

print(f"Inputs:")
for i, col in enumerate(caracteristicas):
    print(f"  {col}: {ejemplo_X[0,i]}")
print(f"Predicción: {prediccion_rf[0]:.2f} MW")


