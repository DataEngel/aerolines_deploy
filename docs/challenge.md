# Parte 1: Elige el mejor modelo según tu criterio y justifica tu elección.

## ¿Por qué XGBoost es mejor que la Regresión Logística?

### 1. Mejor rendimiento en datos no lineales  
XGBoost usa árboles de decisión en boosting, lo que le permite capturar relaciones complejas en los datos.  
La Regresión Logística asume que las variables tienen una relación lineal, lo que limita su capacidad predictiva.

### 2. Manejo del desbalance de clases  
XGBoost permite ajustar el parámetro `scale_pos_weight`, que mejora la detección de vuelos retrasados (clase 1).  
La Regresión Logística con `class_weight` mejora un poco el balance, pero no tanto como XGBoost.

### 3. Mejor capacidad de generalización  
XGBoost reduce el overfitting gracias a técnicas como regularización y pruning de árboles.  
La Regresión Logística puede sufrir con datos no balanceados o con muchas variables categóricas.

### 4. Feature Importance y Selección Automática de Variables  
XGBoost permite visualizar qué variables son más importantes (`plot_importance`).  
La Regresión Logística no tiene un método nativo para esto y requiere más procesamiento manual.

### Conclusión  
XGBoost ofrece mejor precisión, recall y capacidad para detectar retrasos, especialmente en datos desbalanceados y no lineales.

