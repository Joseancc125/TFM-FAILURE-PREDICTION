# TFM-FAILURE-PREDICTION

Sistema de mantenimiento predictivo para equipos industriales basado en machine learning. El proyecto transforma lecturas de sensores en una visión ejecutiva del riesgo de fallo, con análisis técnico, interpretabilidad y dashboard interactivo.

## Memoria técnica

Memoria técnica del proyecto en `memoria_tecnica.md`.

## Guía rápida de arranque

Antes de ejecutar el proyecto, revisa la guía detallada en `guia_de_ejecucion.md`.

## Qué resuelve

- anticipa fallos antes de que se produzcan;
- identifica señales de comportamiento anómalo;
- compara varios modelos predictivos;
- explica qué variables influyen más en el riesgo;
- presenta el resultado en una interfaz clara para negocio y mantenimiento.

## Resultado visual

![Importancia global de variables](reports/figures/phase6/feature_importance.png)

![Explicabilidad SHAP](reports/figures/phase6/shap_importance.png)

## Estructura del proyecto

- `data/raw`: datos originales;
- `data/processed`: datos transformados y artefactos intermedios;
- `src/eda`: análisis exploratorio;
- `src/preprocessing`: limpieza y preparación;
- `src/features`: ingeniería de características;
- `src/anomaly`: detección de anomalías;
- `src/models`: modelos supervisados;
- `src/interpretability`: interpretabilidad;
- `dashboard`: interfaz ejecutiva;
- `models`: modelos entrenados;
- `reports`: resultados, métricas y figuras.

## Cómo ejecutar el proyecto

1. Activa el entorno virtual.
2. Instala dependencias.
3. Ejecuta las fases en orden.
4. Abre el dashboard al final.

### Activar entorno virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución por fases

### Fase 1. Análisis exploratorio

```powershell
python -m src.eda.phase1_eda
```

### Fase 2. Preparación de datos

```powershell
python -m src.preprocessing.phase2_prepare_data
```

### Fase 3. Ingeniería de características

```powershell
python -m src.features.phase3_feature_engineering
```

### Fase 4. Detección de anomalías

```powershell
python -m src.anomaly.phase4_anomaly_detection
```

### Fase 5. Predicción de fallos

```powershell
python -m src.models.phase5_supervised_modeling
```

### Fase 6. Interpretabilidad

```powershell
python -m src.interpretability.phase6_interpretability
```

### Fase 7. Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

## Salidas principales

- `data/processed/supervised_test_predictions.csv`;
- `reports/phase5/supervised_model_comparison.csv`;
- `reports/phase6/interpretability_summary.csv`;
- `reports/figures/phase6/feature_importance.png`;
- `reports/figures/phase6/shap_importance.png`.

## Visión ejecutiva

El modelo final recomendado es `Random Forest`, y el dashboard resume la situación en términos de riesgo, salud de la máquina y señales que requieren atención.

## Dashboard

El panel está orientado a una lectura rápida por parte de responsables de operación y mantenimiento.

- resumen ejecutivo;
- riesgo estimado;
- modelo recomendado;
- factores más influyentes;
- anomalías;
- salud operativa.

## Notas finales

Si cambias el dataset o regeneras artefactos, vuelve a ejecutar las fases desde el inicio para mantener coherencia entre los resultados, las figuras y el dashboard.
