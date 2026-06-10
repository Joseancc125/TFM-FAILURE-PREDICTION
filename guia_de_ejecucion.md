# Guía de ejecución del proyecto

Esta guía explica cómo preparar el entorno y ejecutar cada fase del proyecto de monitorización y predicción de fallos.

## 1. Requisitos previos

- Windows 10 o superior.
- Python instalado.
- Recomendado: entorno virtual `.venv`.
- Archivo de datos `data.csv` disponible en la raíz del proyecto.

## 2. Activación del entorno virtual

Desde PowerShell, situado en la raíz del proyecto:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si el entorno virtual no existe todavía, créalo antes:

```powershell
python -m venv .venv
```

## 3. Instalación de dependencias

Instala las librerías necesarias desde `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Orden recomendado de ejecución

Ejecuta las fases en este orden para reproducir el proyecto completo:

1. análisis exploratorio;
2. preparación de datos;
3. ingeniería de características;
4. detección de anomalías;
5. predicción supervisada;
6. interpretabilidad;
7. dashboard.

## 5. Ejecución por fases

### Fase 1. Análisis exploratorio

```powershell
python -m src.eda.phase1_eda
```

Salida esperada:

- tablas de resumen en `reports/phase1/`;
- gráficos exploratorios asociados al dataset.

### Fase 2. Preparación de datos

```powershell
python -m src.preprocessing.phase2_prepare_data
```

Salida esperada:

- `data/processed/clean_dataset.csv`
- `data/processed/train_dataset.csv`
- `data/processed/test_dataset.csv`
- `data/processed/preprocessor.joblib`
- `reports/phase2/data_preparation_report.csv`

### Fase 3. Ingeniería de características

```powershell
python -m src.features.phase3_feature_engineering
```

Salida esperada:

- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `data/processed/feature_engineered_dataset.csv`
- `data/processed/risk_scaler.joblib`
- `data/processed/health_thresholds.json`
- `reports/phase3/feature_engineering_report.csv`

### Fase 4. Detección de anomalías

```powershell
python -m src.anomaly.phase4_anomaly_detection
```

Salida esperada:

- `data/processed/train_anomaly_scores.csv`
- `data/processed/test_anomaly_scores.csv`
- `data/processed/isolation_forest_model.joblib`
- `data/processed/lof_model.joblib`
- `reports/phase4/anomaly_detection_report.csv`

### Fase 5. Predicción de fallos

```powershell
python -m src.models.phase5_supervised_modeling
```

Salida esperada:

- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/lightgbm.joblib`
- `models/supervised_models_metadata.json`
- `data/processed/supervised_test_predictions.csv`
- `reports/phase5/supervised_model_comparison.csv`

### Fase 6. Interpretabilidad

```powershell
python -m src.interpretability.phase6_interpretability
```

Salida esperada:

- `reports/phase6/feature_importance.csv`
- `reports/phase6/shap_importance.csv`
- `reports/phase6/interpretability_summary.csv`
- `reports/figures/phase6/feature_importance.png`
- `reports/figures/phase6/shap_importance.png`

### Fase 7. Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

El dashboard se abrirá normalmente en `http://localhost:8501`.

## 6. Verificación rápida

Si quieres comprobar que todo está correcto tras ejecutar las fases, revisa que existan los siguientes archivos clave:

- `reports/phase5/supervised_model_comparison.csv`
- `reports/phase6/interpretability_summary.csv`
- `data/processed/supervised_test_predictions.csv`
- `data/processed/test_anomaly_scores.csv`

## 7. Recomendaciones prácticas

- Ejecuta las fases en orden, ya que cada una depende de las salidas de la anterior.
- Si cambias el dataset, vuelve a ejecutar desde la fase 1 o, como mínimo, desde la fase 2.
- Antes de abrir el dashboard, asegúrate de haber generado las salidas de las fases 5 y 6.
- Si el dashboard no arranca, verifica que el entorno virtual esté activado y que `streamlit` esté instalado.

## 8. Comando útil de comprobación

Para validar rápidamente que el archivo del dashboard no tiene errores de sintaxis:

```powershell
python -m py_compile dashboard/app.py
```

## 9. Observación final

La guía está pensada para reproducir la solución completa con el menor número de pasos posible y sin depender de herramientas externas adicionales.
