# Memoria técnica del Trabajo Fin de Máster

## 1. Introducción

La digitalización industrial ha impulsado la instrumentación de máquinas y equipos mediante sensores capaces de registrar, de forma continua, variables operativas relevantes para la supervisión del estado de los activos. Este contexto abre la puerta al uso de técnicas de analítica avanzada y aprendizaje automático para anticipar degradaciones, detectar comportamientos anómalos y apoyar la toma de decisiones en mantenimiento.

En este Trabajo Fin de Máster se ha desarrollado una solución de Inteligencia Artificial aplicada a un conjunto de datos industrial proporcionado por el tutor, cuyo objetivo es determinar si una máquina presenta un fallo (`fail`) a partir de lecturas de distintos sensores. El proyecto no se limita a construir un clasificador, sino que integra análisis exploratorio, preparación de datos, ingeniería de características, detección de anomalías, modelado supervisado, interpretabilidad y visualización ejecutiva.

## 2. Objetivo y alcance

El objetivo general del trabajo es diseñar e implementar un sistema inteligente capaz de analizar el estado operativo de los equipos y generar alertas tempranas de riesgo con utilidad práctica para un entorno de negocio y mantenimiento industrial.

De forma específica, la solución persigue:

- analizar el comportamiento de las señales disponibles;
- detectar patrones anómalos de funcionamiento;
- predecir la probabilidad de fallo de la máquina;
- comparar distintos algoritmos de Machine Learning;
- interpretar los factores con mayor influencia en la predicción;
- presentar los resultados en un dashboard comprensible para usuarios no técnicos.

El alcance del proyecto se ha mantenido alineado con un enfoque realista de TFM: una solución completa y funcional sobre un dataset acotado, con foco en la trazabilidad técnica y la utilidad operativa.

## 3. Dataset y significado de las variables

El dataset contiene 944 registros con las variables siguientes:

- `footfall`: número de personas u objetos cercanos a la máquina;
- `tempMode`: modo de funcionamiento térmico;
- `AQ`: índice de calidad del aire;
- `USS`: sensor ultrasónico;
- `CS`: sensor de corriente eléctrica;
- `VOC`: nivel de compuestos orgánicos volátiles;
- `RP`: posición rotacional o RPM;
- `IP`: presión de entrada;
- `Temperature`: temperatura de operación;
- `fail`: indicador binario de fallo.

Desde una perspectiva industrial, estas señales reflejan condiciones de entorno, carga de funcionamiento, comportamiento térmico y comportamiento eléctrico/mecánico. La hipótesis de trabajo es que la combinación de estas medidas permite identificar estados previos al fallo y no únicamente el fallo una vez producido.

## 4. Fase 1. Comprensión del problema y análisis exploratorio

Se realizó un análisis exploratorio inicial para comprender la estructura del conjunto de datos, la calidad de la información y la distribución de la variable objetivo.

### Resultados principales

- Registros originales: 944.
- Clases de la variable objetivo:
  - clase 0: 551 registros, 58.37%;
  - clase 1: 393 registros, 41.63%.
- Valores nulos: 0.
- Registros duplicados detectados: 1.

### Interpretación técnica y de negocio

La exploración confirmó que el problema no está condicionado por una ausencia de información, sino por la necesidad de extraer señal operativa útil a partir de variables de sensores. La distribución de clases, aunque no extremadamente desbalanceada, justifica el uso de métricas sensibles a la clase positiva, especialmente `recall`, `precision`, `F1` y `PR-AUC`, dado que en mantenimiento industrial un fallo no detectado puede tener un coste mayor que un falso positivo.

Desde el punto de vista de negocio, el evento relevante es `fail`. El valor de la solución no depende únicamente de acertar globalmente, sino de detectar con antelación los casos de mayor riesgo y reducir paradas no planificadas.

## 5. Fase 2. Calidad y preparación de los datos

### Resumen de calidad

- Registros originales: 944.
- Registros limpios: 943.
- Duplicados eliminados: 1.
- Valores nulos: 0.

### Decisiones adoptadas

- eliminación del registro duplicado;
- partición estratificada en entrenamiento y prueba;
- aplicación de un preprocesamiento consistente y reutilizable.

### Justificación

No fue necesaria imputación de valores faltantes, ya que el conjunto de datos estaba completo. La eliminación del duplicado fue suficiente para depurar la base. La división estratificada garantizó que la proporción de fallos se mantuviera estable entre entrenamiento y prueba, evitando una evaluación artificialmente optimista o pesimista.

La fase generó 754 observaciones de entrenamiento y 189 de prueba, con tasas de fallo muy similares entre subconjuntos:

- train: 41.64%;
- test: 41.80%;
- global: 41.68%.

Esta consistencia refuerza la validez de las métricas obtenidas en las etapas posteriores.

## 6. Fase 3. Ingeniería de características

Con el objetivo de capturar relaciones más informativas entre sensores, se generaron 11 variables derivadas a partir de transformaciones y combinaciones de las señales originales.

### Variables destacadas

- `risk_score`;
- `machine_health_index`;
- `air_pressure_ratio`;
- `air_risk`;
- `sensor_gap`;
- `sensor_stability`;
- `operating_intensity`;
- `mechanical_pressure`;
- `control_balance`;
- `thermal_load`.

### Criterio de diseño

La ingeniería de características se diseñó con una lógica de dominio industrial: representar el estado de la máquina no solo a partir de lecturas individuales, sino mediante indicadores agregados que sinteticen riesgo, carga operativa y salud relativa del equipo.

Por ejemplo:

- `machine_health_index` permite traducir múltiples señales a una escala operativa comprensible;
- `risk_score` condensa el riesgo agregado en un indicador único;
- ratios y combinaciones entre presión, temperatura y señales eléctricas permiten capturar estados de carga o degradación difíciles de apreciar con variables aisladas.

### Validación posterior

La interpretabilidad del modelo confirmó la utilidad de estas transformaciones: `risk_score` y `machine_health_index` aparecieron entre las variables más influyentes, lo que respalda que la información derivada aportó valor real al proceso de predicción.

## 7. Fase 4. Detección de anomalías

Se incorporaron dos técnicas de detección de anomalías no supervisadas:

- Isolation Forest;
- Local Outlier Factor (LOF).

### Resultados obtenidos

| Modelo | ROC-AUC test | PR-AUC test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.5206 | 0.4455 | 0.4231 | 0.1392 | 0.2095 |
| LOF | 0.5341 | 0.4706 | 0.5000 | 0.1519 | 0.2330 |

### Valor metodológico

Aunque el rendimiento de estas técnicas es inferior al de los modelos supervisados, su papel es relevante como capa complementaria. En escenarios industriales, una anomalía no implica necesariamente un fallo inminente, pero sí puede señalar una desviación de comportamiento que merece atención.

LOF mostró un rendimiento ligeramente superior a Isolation Forest, lo que sugiere que la estructura local del espacio de variables contiene señal útil. Aun así, por sí sola, esta información no resulta suficiente para una predicción robusta del fallo.

## 8. Fase 5. Predicción de fallos

Se compararon tres modelos supervisados:

- Random Forest;
- XGBoost;
- LightGBM.

### Resultados en el conjunto de prueba

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9732 | 0.9570 | 0.8736 | 0.9620 | 0.9157 |
| XGBoost | 0.9659 | 0.9499 | 0.8409 | 0.9367 | 0.8862 |
| LightGBM | 0.9611 | 0.9393 | 0.8571 | 0.9114 | 0.8834 |

### Elección del modelo final

El modelo seleccionado fue **Random Forest**, al presentar el mejor equilibrio entre capacidad discriminativa, sensibilidad ante fallos y estabilidad general en prueba.

### Criterios de selección

- mejor `F1` en test;
- mayor `ROC-AUC`;
- `recall` muy elevado, esencial para minimizar fallos no detectados;
- comportamiento consistente y fácil de explicar mediante importancia de variables.

### Umbral de decisión

El umbral óptimo seleccionado fue 0.4375. Este valor se eligió buscando maximizar el equilibrio entre precisión y recall, con prioridad operativa en la detección temprana de fallos.

### Implicación operativa

El modelo final permite asignar una probabilidad de fallo a cada registro y utilizar dicha probabilidad para priorizar inspecciones, programar mantenimiento preventivo y reducir el riesgo de paradas inesperadas.

## 9. Fase 6. Interpretación de resultados

Se aplicaron dos técnicas de interpretabilidad:

- Feature Importance;
- SHAP.

### Variables con mayor influencia

Las variables más relevantes fueron:

- `risk_score`;
- `machine_health_index`;
- `VOC`;
- `air_pressure_ratio`;
- `air_risk`;
- `sensor_gap`.

### Lectura técnica

La interpretación confirma que las variables derivadas concentran una parte importante de la capacidad predictiva. Esto refuerza la idea de que la incorporación de conocimiento de dominio en forma de indicadores sintéticos mejora tanto el rendimiento como la explicabilidad.

Las señales relacionadas con riesgo agregado y salud operativa resultaron más informativas que varios sensores tomados de forma independiente, lo cual es coherente con un escenario industrial en el que la degradación suele manifestarse como combinación de síntomas.

## 10. Fase 7. Visualización y dashboard

Se desarrolló un dashboard interactivo con Streamlit orientado a perfiles no técnicos y a usuarios de negocio.

### Elementos incluidos

- resumen ejecutivo del estado de la máquina;
- recomendación del modelo final;
- indicadores principales de riesgo;
- comparativa técnica de modelos en una vista secundaria;
- distribución del riesgo estimado;
- impacto operativo de las predicciones;
- variables más influyentes;
- anomalías detectadas;
- indicador ejecutivo de salud de la máquina.

### Criterios de diseño

La interfaz prioriza la comprensión y la toma de decisiones. Por ello, el lenguaje visual y textual se ha adaptado a una audiencia de mantenimiento, operaciones o dirección, manteniendo el detalle técnico en secciones plegables o de consulta secundaria.

## 11. Limitaciones

El trabajo presenta las limitaciones propias de un TFM con un dataset acotado:

- el tamaño del conjunto de datos es limitado para una validación de generalización amplia;
- la variable `fail` no incorpora información temporal del proceso de degradación;
- la capa de anomalías complementa, pero no sustituye, la predicción supervisada;
- no se ha desplegado una solución completa de MLOps, monitorización continua ni API de producción.

## 12. Conclusiones

El modelo Random Forest se posiciona como la mejor alternativa final por rendimiento y equilibrio operativo. Además, la combinación de `risk_score`, `machine_health_index` y las señales derivadas demuestra que el conocimiento de dominio mejora de forma clara la calidad del sistema.

## 13. Mejoras futuras

Como líneas de evolución se proponen las siguientes:

- ampliar el histórico de datos para validar robustez;
- incorporar dimensión temporal a las variables de proceso;
- calibrar probabilidades y umbrales por criticidad del activo;
- añadir alertas automáticas y monitorización continua;
- desplegar el sistema mediante API o plataforma de explotación;
- incorporar herramientas MLOps como MLflow o Docker.
