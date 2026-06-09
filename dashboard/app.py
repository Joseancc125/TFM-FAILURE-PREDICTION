from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="TFM | Riesgo de fallo de máquina",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1rem;
        }
        .hero-box h1 { font-size: 2rem; margin-bottom: 0.25rem; }
        .hero-box p { margin-bottom: 0; color: #e2e8f0; }
        .mini-note {
            color: #475569;
            font-size: 0.92rem;
            margin-top: -0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PHASE5_COMPARISON = ROOT / "reports" / "phase5" / "supervised_model_comparison.csv"
PHASE6_FEATURE = ROOT / "reports" / "phase6" / "feature_importance.csv"
PHASE6_SHAP = ROOT / "reports" / "phase6" / "shap_importance.csv"
PHASE6_SUMMARY = ROOT / "reports" / "phase6" / "interpretability_summary.csv"
TEST_PREDICTIONS = ROOT / "data" / "processed" / "supervised_test_predictions.csv"
TEST_FEATURES = ROOT / "data" / "processed" / "test_features.csv"
TEST_ANOMALY = ROOT / "data" / "processed" / "test_anomaly_scores.csv"

MODEL_LABELS = {
    "random_forest": "Opción recomendada",
    "xgboost": "Opción alternativa 1",
    "lightgbm": "Opción alternativa 2",
}


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "comparison": load_csv(PHASE5_COMPARISON),
        "feature_importance": load_csv(PHASE6_FEATURE),
        "shap_importance": load_csv(PHASE6_SHAP),
        "summary": load_csv(PHASE6_SUMMARY),
        "predictions": load_csv(TEST_PREDICTIONS),
        "test_features": load_csv(TEST_FEATURES),
        "test_anomaly": load_csv(TEST_ANOMALY),
    }


def get_best_model_name(comparison: pd.DataFrame) -> str:
    best_row = comparison[comparison["split"] == "test"].sort_values(["f1", "pr_auc", "recall"], ascending=False).iloc[0]
    return str(best_row["model"])


def friendly_model_name(model_name: str) -> str:
    return f"{MODEL_LABELS.get(model_name, model_name)} · {model_name}"


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def make_confusion_figure(tp: int, tn: int, fp: int, fn: int, title: str) -> go.Figure:
    matrix = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicción", y="Caso real", color="Casos"),
        x=["Sin fallo", "Fallo"],
        y=["Sin fallo", "Fallo"],
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False, height=360)
    return fig


def model_prediction_columns(predictions: pd.DataFrame, model_name: str) -> tuple[str, str]:
    probability_column = f"{model_name}_probability"
    prediction_column = f"{model_name}_prediction"
    if probability_column not in predictions.columns or prediction_column not in predictions.columns:
        raise KeyError(f"Missing prediction columns for model: {model_name}")
    return probability_column, prediction_column


def main() -> None:
    artifacts = load_artifacts()
    comparison = artifacts["comparison"]
    predictions = artifacts["predictions"]
    test_features = artifacts["test_features"]
    test_anomaly = artifacts["test_anomaly"]
    feature_importance = artifacts["feature_importance"]
    shap_importance = artifacts["shap_importance"]
    summary = artifacts["summary"]

    best_model_name = get_best_model_name(comparison)
    model_options = comparison[comparison["split"] == "test"]["model"].drop_duplicates().tolist()
    display_options = [friendly_model_name(model) for model in model_options]
    selected_display = st.sidebar.selectbox("Qué modelo quieres revisar", display_options, index=model_options.index(best_model_name))
    selected_model = model_options[display_options.index(selected_display)]

    probability_column, prediction_column = model_prediction_columns(predictions, selected_model)
    selected_metrics = comparison[(comparison["model"] == selected_model) & (comparison["split"] == "test")].iloc[0]
    y_true = predictions["fail"].astype(int)
    y_pred = predictions[prediction_column].astype(int)
    y_prob = predictions[probability_column].astype(float)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    anomaly_columns = [column for column in test_anomaly.columns if column.endswith("_anomaly_flag")]
    anomaly_summary = pd.DataFrame(
        {
            "Método": [column.replace("_anomaly_flag", "") for column in anomaly_columns],
            "Tasa de alertas": [test_anomaly[column].mean() * 100 for column in anomaly_columns],
        }
    ).sort_values("Tasa de alertas", ascending=False)
    anomaly_summary["Tasa de alertas"] = anomaly_summary["Tasa de alertas"].map(lambda value: f"{value:.2f}%")

    st.markdown(
        """
        <div class="hero-box">
            <h1>Estado de la máquina y riesgo de fallo</h1>
            <p>Vista pensada para operación y mantenimiento: qué tan probable es un fallo, qué señales se están deteriorando y dónde conviene actuar primero.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='mini-note'>El panel resume el estado en lenguaje operativo, sin necesidad de conocer la parte técnica del modelo.</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "Un valor alto de riesgo significa que la máquina se parece más a los casos que terminaron en fallo. El objetivo es anticiparse y priorizar inspecciones.",
        icon="ℹ️",
    )

    best_metrics = comparison[(comparison["model"] == best_model_name) & (comparison["split"] == "test")].iloc[0]
    top_features = feature_importance.sort_values("importance", ascending=False).head(3)["feature"].tolist()

    st.subheader("Resumen ejecutivo")
    exec_left, exec_mid, exec_right = st.columns([1.1, 1, 1.2])
    with exec_left:
        metric_card("Modelo recomendado", MODEL_LABELS.get(best_model_name, best_model_name))
    with exec_mid:
        metric_card("Éxito de detección", f"{best_metrics['recall']:.3f}")
    with exec_right:
        st.markdown(
            f"""
            <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:14px;padding:0.95rem 1rem;'>
                <div style='font-weight:700;color:#1d4ed8;margin-bottom:0.25rem;'>Conclusión</div>
                <div style='color:#1e293b;font-size:0.94rem;line-height:1.45;'>
                    El sistema recomienda priorizar el modelo <b>{friendly_model_name(best_model_name)}</b> porque ofrece el mejor equilibrio entre detectar fallos y mantener una tasa baja de errores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p class='mini-note'>Los factores que más empujan la alerta son: <b>{', '.join(top_features)}</b>.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Por qué esto importa para el negocio")
    business_cols = st.columns(3)
    business_cards = [
        (
            "Menos paradas no planificadas",
            "Anticipar el riesgo permite intervenir antes de que la máquina falle y afecte la producción.",
        ),
        (
            "Mantenimiento mejor priorizado",
            "Las señales del modelo ayudan a centrar revisiones en los equipos con mayor deterioro operativo.",
        ),
        (
            "Decisiones más homogéneas",
            "El sistema aporta un criterio cuantitativo para decidir cuándo revisar, vigilar o escalar una incidencia.",
        ),
    ]
    for column, (title, body) in zip(business_cols, business_cards, strict=False):
        with column:
            st.markdown(
                f"""
                <div style='background:#ffffff;border:1px solid #dbe4f0;border-radius:14px;padding:1rem 1rem 0.9rem 1rem;height:100%;'>
                    <div style='font-weight:700;font-size:0.98rem;margin-bottom:0.35rem;color:#0f172a;'>{title}</div>
                    <div style='font-size:0.92rem;line-height:1.45;color:#475569;'>{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Antes de ver los resultados")
    story_cols = st.columns(4)
    story_cards = [
        (
            "Qué datos tenemos",
            "Medidas de sensores, estado operativo y señales históricas de máquinas que han trabajado en condiciones normales y en fallo.",
        ),
        (
            "Cuál es el problema",
            "Detectar con antelación qué equipos tienen más papeletas de fallar para poder actuar antes de que la avería ocurra.",
        ),
        (
            "Qué buscamos",
            "Priorizar revisiones, reducir paradas no planificadas y ayudar a mantenimiento a enfocarse en los equipos más críticos.",
        ),
        (
            "Qué modelos entrenamos",
            "Probamos varios enfoques automáticos y nos quedamos con el que mejor separa los casos normales de los casos de fallo.",
        ),
    ]
    for column, (title, body) in zip(story_cols, story_cards, strict=False):
        with column:
            st.markdown(
                f"""
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:1rem 1rem 0.9rem 1rem;height:100%;'>
                    <div style='font-weight:700;font-size:1rem;margin-bottom:0.4rem;color:#0f172a;'>{title}</div>
                    <div style='font-size:0.92rem;line-height:1.45;color:#334155;'>{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Riesgo medio estimado", f"{y_prob.mean() * 100:.1f}%")
    with col2:
        metric_card("Fallos detectados", f"{selected_metrics['recall']:.3f}")
    with col3:
        metric_card("Calidad global", f"{selected_metrics['f1']:.3f}")
    with col4:
        metric_card("Punto de aviso", f"{selected_metrics['threshold']:.3f}")

    with st.expander("Ver detalle técnico de los modelos", expanded=False):
        detail_left, detail_right = st.columns([1.3, 1])
        with detail_left:
            st.subheader("Comparativa de modelos")
            comparison_view = comparison[comparison["split"] == "test"].sort_values(["f1", "pr_auc", "recall"], ascending=False).copy()
            comparison_view = comparison_view.rename(
                columns={
                    "model": "Modelo",
                    "split": "Fase",
                    "precision": "Precisión",
                    "recall": "Fallos detectados",
                    "f1": "Calidad global",
                    "pr_auc": "Separación de casos",
                    "roc_auc": "Poder de discriminación",
                    "threshold": "Punto de aviso",
                }
            )
            st.dataframe(comparison_view, use_container_width=True, hide_index=True)
        with detail_right:
            st.subheader("Señales que explican el riesgo")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    chart_left, chart_right = st.columns([1.1, 1])
    with chart_left:
        st.subheader("Distribución del riesgo de fallo")
        prob_fig = px.histogram(
            predictions,
            x=probability_column,
            color=predictions["fail"].map({0: "Sin fallo", 1: "Con fallo"}),
            barmode="overlay",
            nbins=25,
            title="Dónde se concentra el riesgo estimado",
            labels={"color": "Caso real", "x": "Probabilidad de fallo"},
            opacity=0.75,
        )
        prob_fig.update_layout(legend_title_text="Caso real", height=360)
        st.plotly_chart(prob_fig, use_container_width=True)
    with chart_right:
        st.subheader("Impacto operativo de las predicciones")
        st.plotly_chart(make_confusion_figure(tp, tn, fp, fn, "Resultado del modelo seleccionado"), use_container_width=True)

    st.divider()
    bottom_left, bottom_right = st.columns([1, 1])
    with bottom_left:
        st.subheader("Principales palancas del riesgo")
        importance_fig = px.bar(
            feature_importance.sort_values("importance", ascending=True).tail(12),
            x="importance",
            y="feature",
            orientation="h",
            title="Qué sensores mueven más la predicción",
        )
        importance_fig.update_layout(height=420)
        st.plotly_chart(importance_fig, use_container_width=True)
    with bottom_right:
        st.subheader("Cómo interpreta el modelo el riesgo")
        shap_fig = px.bar(
            shap_importance.sort_values("mean_abs_shap", ascending=True).tail(12),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Peso explicativo de cada señal",
        )
        shap_fig.update_layout(height=420)
        st.plotly_chart(shap_fig, use_container_width=True)

    st.caption("La parte técnica queda aquí debajo para quien quiera revisar el detalle; arriba se mantiene una lectura más ejecutiva.")

    st.divider()
    anomaly_left, anomaly_right = st.columns([1, 1])
    with anomaly_left:
        st.subheader("Alertas de comportamiento anómalo")
        st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)
    with anomaly_right:
        st.subheader("Indicador ejecutivo de salud de la máquina")
        health_fig = px.histogram(
            test_features,
            x="machine_health_level",
            category_orders={
                "machine_health_level": [
                    "Level 1 - Healthy",
                    "Level 2 - Low Risk",
                    "Level 3 - Moderate Risk",
                    "Level 4 - High Risk",
                    "Level 5 - Critical",
                ]
            },
            title="Cómo se reparte el estado operativo de los equipos",
        )
        health_fig.update_layout(height=360, xaxis_title="Nivel", yaxis_title="Registros")
        st.plotly_chart(health_fig, use_container_width=True)

    st.divider()
    st.subheader("Ver un caso concreto")
    row_index = st.slider("Elige una muestra del conjunto de prueba", 0, len(test_features) - 1, 0)
    selected_row = test_features.iloc[row_index].to_frame().T
    selected_view = pd.DataFrame(
        {
            "Campo": selected_row.columns,
            "Valor": selected_row.iloc[0].values,
        }
    )
    st.dataframe(selected_view, use_container_width=True, hide_index=True)

    st.caption(
        f"Modelo visualizado: {selected_display} | Recomendado: {friendly_model_name(best_model_name)} | Casos analizados: {len(test_features)}"
    )


if __name__ == "__main__":
    main()
