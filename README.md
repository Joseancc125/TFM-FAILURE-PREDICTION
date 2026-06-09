# TFM-FAILURE-PREDICTION

Proyecto de TFM para monitorización y predicción de fallos en equipos industriales mediante machine learning.

## Estructura inicial

- `data/raw`: datos originales
- `data/processed`: datos transformados
- `src/eda`: análisis exploratorio y validación inicial
- `src/features`: ingeniería de características
- `src/models`: entrenamiento de modelos
- `src/evaluation`: métricas y comparación de modelos
- `src/visualization`: gráficos y utilidades visuales
- `dashboard`: aplicación Streamlit o Dash
- `models`: artefactos serializados
- `reports`: resultados, figuras y documentación parcial
- `tests`: pruebas básicas

## Fase 1

El punto de entrada inicial está en `src/eda/phase1_eda.py`.

### Ejecución

Desde la raíz del proyecto:

```bash
python -m src.eda.phase1_eda
```

## Fase 2

La preparación y limpieza de datos está en `src/preprocessing/phase2_prepare_data.py`.

### Ejecución

```bash
python -m src.preprocessing.phase2_prepare_data
```

### Salidas generadas

- `data/processed/clean_dataset.csv`
- `data/processed/train_dataset.csv`
- `data/processed/test_dataset.csv`
- `data/processed/preprocessor.joblib`
- `reports/phase2/data_preparation_report.csv`

## Fase 3

La ingeniería de características está en `src/features/phase3_feature_engineering.py`.

### Ejecución

```bash
python -m src.features.phase3_feature_engineering
```

### Salidas generadas

- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `data/processed/feature_engineered_dataset.csv`
- `data/processed/risk_scaler.joblib`
- `data/processed/health_thresholds.json`
- `reports/phase3/feature_engineering_report.csv`

## Fase 4

La detección de anomalías está en `src/anomaly/phase4_anomaly_detection.py`.

### Ejecución

```bash
python -m src.anomaly.phase4_anomaly_detection
```

### Salidas generadas

- `data/processed/train_anomaly_scores.csv`
- `data/processed/test_anomaly_scores.csv`
- `data/processed/isolation_forest_model.joblib`
- `data/processed/lof_model.joblib`
- `reports/phase4/anomaly_detection_report.csv`

## Fase 5

La predicción supervisada está en `src/models/phase5_supervised_modeling.py`.

### Ejecución

```bash
python -m src.models.phase5_supervised_modeling
```

### Salidas generadas

- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/lightgbm.joblib`
- `models/supervised_models_metadata.json`
- `data/processed/supervised_test_predictions.csv`
- `reports/phase5/supervised_model_comparison.csv`

## Fase 6

La interpretabilidad está en `src/interpretability/phase6_interpretability.py`.

### Ejecución

```bash
python -m src.interpretability.phase6_interpretability
```

### Salidas generadas

- `reports/phase6/feature_importance.csv`
- `reports/phase6/shap_importance.csv`
- `reports/phase6/interpretability_summary.csv`
- `reports/figures/phase6/feature_importance.png`
- `reports/figures/phase6/shap_importance.png`

## Fase 7

El dashboard está en `dashboard/app.py`.

### Ejecución

```bash
streamlit run dashboard/app.py
```

### Incluye

- Comparativa de modelos supervisados
- Métricas del mejor modelo
- Distribución de probabilidades de fallo
- Matriz de confusión
- Importancia global y `SHAP`
- Anomalías detectadas
- Índice de salud de la máquina