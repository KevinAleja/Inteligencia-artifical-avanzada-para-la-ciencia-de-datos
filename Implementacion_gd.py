"""
===============================================================================
Activar Enviroment:
source IA_AVANZADA/bin/activate
===============================================================================
Codigo que implementa una regresión lineal para el calculo del
#n de miliwatts/hr de una planta de energía de ciclo combinado


NOTA: Si se quiere ver cambio entre más o menos features, descomentar/comentar las 
lineas 56, 165, 233

Kevin Alejandro Ramírez Luna | A01711063@tec.mx
===============================================================================
"""


# Librerias a utilizar 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns

# Configuración para reproducibilidad
np.random.seed(42)

# 1. Carga y preparación de datos
print("Cargando el dataset...\n")

# Cambio del nombre de las columnas a las que habia en la documentación
nombres_columnas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure',
                    'Relative_humidity', 'Net_hourly_electrical_energy']

# Lectura del archivo .xlsx
df = pd.read_excel('/home/kevinalejarmzl/Documentos/Inteligencia_Artifical_Avanzada/modulo_2_aprendizaje_maquina/Folds5x2_pp.xlsx', header=0, names=nombres_columnas)

# Limpieza de datos
df = df.dropna() # Eliminamos algún valor nulo si es que lo hubiera 
print(f"Dataset cargado: {df.shape[0]} instancias, {df.shape[1]} features")


# 2. Matriz de correlación para ver relaciones entre variables y ver cuales son utiles en nuestra regresión
print("\n=== MATRIZ DE CORRELACIÓN ===")
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", center=0)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()

# 3. Selección de variables
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
#X = df[['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']].values
X = df[['Temperature', 'Exhaust_vacuum']].values

y = df['Net_hourly_electrical_energy'].values.reshape(-1, 1) # Colocamos el reshape para trasformar una lista a una matriz (n,1)

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 4. Escalado de características 
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

y_original = y.flatten()  # Solo aplanamos para que tenga shape (n,)

print("\n=== Despues del escalamiento===")
print("X - Medias:", scaler_X.mean_)
print("X - Escalas:", scaler_X.scale_)
print("y - (primeros valores):", y_original[:5])

# 5. División de datos para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_original, test_size=0.2, random_state=42
)

print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Prueba: {X_test.shape[0]} muestras")

# 6. Funciones del modelo
def hipotesis(X, theta, b):
    """y = b + theta * X"""
    return b + (X @ theta) # Multiplición entre matrices

def MSE(X, y, theta, b):
    """Error Cuadrático Medio"""
    m = len(y)
    y_predicha = hipotesis(X, theta, b)
    error = y_predicha - y
    return np.mean(error ** 2)

def calcular_gradientes(X_batch, y_batch, theta, b):
    """Cálculo de gradientes"""
    m_batch = len(X_batch)
    predicciones = hipotesis(X_batch, theta, b)
    error = predicciones - y_batch
    
    # Gradiente para bias
    grad_b = np.mean(error)
    
    # Gradientes para cada theta (característica)
    grad_theta = np.zeros_like(theta) #Inicializamos en 0
    for j in range(len(theta)):
        grad_theta[j] = np.mean(error * X_batch[:, j])
    
    return grad_theta, grad_b

def GD(X, y, theta_inicial, b_inicial, a, epochs):
    """Implementación del Descenso de Gradiente(GD)"""
    theta = theta_inicial.copy()
    b = b_inicial
    m = len(y)
    historial_error = []
    mejores_parametros = None 
    mejor_error = float('inf')
    
    for epoch in range(epochs):
        # Calcular gradientes con todo el conjunto de datos
        grad_theta, grad_b = calcular_gradientes(X, y, theta, b)
        
        # Actualizar parámetros
        theta -= a * grad_theta #Lerning rate * gradiente
        b -= a * grad_b
        
        # Monitoreo del error
        error_epoch = MSE(X, y, theta, b)
        historial_error.append(error_epoch)
        
        # Guardar mejores parámetros
        if error_epoch < mejor_error:
            mejor_error = error_epoch
            mejores_parametros = (theta.copy(), b)
        
        # Proceso de entrenamiento
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Error = {error_epoch:.6f}")
    
    return mejores_parametros[0], mejores_parametros[1], historial_error

# 7. Entrenamiento del modelo
print("\n=== Entrenamiento del modelo ===")
# Datos iniciales seleccionados de manera arbitraria 
n_caracteristicas = X_train.shape[1] # features a utilizar
theta_inicial = np.zeros(n_caracteristicas) #Vector de 0´s en relacion al numero de x
b_inicial = 0.0
learning_rate = 0.1 # Lerning rate chiquito para que los gradientes convergan relativamente lento
epochs = 1000


# Asignación de los mejores datos para nuestro modelo
theta_entrenado, b_entrenado, historial_error = GD(
    X_train, y_train, theta_inicial, b_inicial,
    learning_rate, epochs
)

# 8. Evaluación
print("\n=== ¡Resultados finales! ===")
print("Parámetros encontrados:")

# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
#caracteristicas = ['Temperature', 'Exhaust_vacuum', 'Ambient_pressure', 'Relative_humidity']
caracteristicas = ['Temperature', 'Exhaust_vacuum']
for i, col in enumerate(caracteristicas):
    print(f"  theta_{col}: {theta_entrenado[i]:.6f}")
print(f"  bias (b): {b_entrenado:.6f}")

# Predicciones
y_pred_train = hipotesis(X_train, theta_entrenado, b_entrenado)
y_pred_test = hipotesis(X_test, theta_entrenado, b_entrenado)

# =============================================================
# Métricas de evaluación}
# =============================================================

# Llamamos a la función del MSE para evaluación de nuestro resultado
mse_train = MSE(X_train, y_train, theta_entrenado, b_entrenado)
mse_test = MSE(X_test, y_test, theta_entrenado, b_entrenado)

# R² para verificación de la linearidad y varianza de datos
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nMétricas de evaluación:")
print(f"Entrenamiento - MSE: {mse_train:.6f}, R^2: {r2_train:.4f}")
print(f"Prueba        - MSE: {mse_test:.6f}, R^2: {r2_test:.4f}")

if r2_test >= 0.5:
    print("\n===== Hay linearidad entre los datos :D  =====")
else:
    print("\n=====  ¡¡¡No hay linearidad entre los datos!!!  =====")

# 9. Visualización
plt.figure(figsize=(15, 5))

# Evolución del error
plt.subplot(1, 3, 1)
plt.plot(historial_error)
plt.title('Evolución del Error (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)

# Predicciones vs Valores reales (train)
plt.subplot(1, 3, 2)
plt.scatter(y_train, y_pred_train, alpha=0.5, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title(f'Train: R² = {r2_train:.3f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones (MW)')
plt.grid(True)

# Predicciones vs Valores reales (test)
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Test: R² = {r2_test:.3f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones (escaladas)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Predicción de ejemplo (desescalada)
print("\n=== PREDICCIÓN DE EJEMPLO ===")

# Valores típicos del dataset
# ¡¡¡¡ Descomentar si es que se quiere trabajar con todos las features !!!!
#ejemplo_X = np.array([[25.0, 60.0, 1013.0, 80.0]])  # Temperature, Exhaust_vacuum, Ambient_pressure, Relative_humidity
ejemplo_X = np.array([[25.0, 60.0]])

ejemplo_X_scaled = scaler_X.transform(ejemplo_X)
prediccion_real = hipotesis(ejemplo_X_scaled, theta_entrenado, b_entrenado) 

print(f"Inputs:")
for i, col in enumerate(caracteristicas):
    print(f"  {col}: {ejemplo_X[0,i]}")
# Convertimos un array a un float usando .item
print(f"Predicción (escalada): {prediccion_real.item():.4f} MW")
print(f"Predicción real: {prediccion_real.item():.2f} MW") 





