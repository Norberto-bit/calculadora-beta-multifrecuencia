import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuración de la página web
st.set_page_config(
    page_title="Calculadora de Beta con Frecuencias - NR Advisory",
    page_icon="📊",
    layout="wide",
)

# Estilo visual moderno y limpio
st.title("📊 Análisis de Rentabilidad, Riesgo y Beta con Frecuencias Múltiples")
st.markdown(
    "Herramienta académica avanzada para evaluar el impacto de la frecuencia"
    " temporal (diaria, semanal, mensual) en el riesgo y el CAPM."
)

# --- PANEL LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Parámetros de Análisis")

# Selección de Tickers
ticker_accion = st.sidebar.text_input(
    "Ticker de la Acción (Ej: AAPL, MELI, GGAL)", value="AAPL"
).upper()
ticker_mercado = st.sidebar.text_input(
    "Ticker de Referencia (Benchmark)", value="SPY"
).upper()

# Selector de Frecuencia de Datos
frecuencia_seleccionada = st.sidebar.selectbox(
    "Frecuencia de Cotizaciones", options=["Diaria", "Semanal", "Mensual"], index=0
)

# Asignar parámetro de Yahoo Finance y factor de anualización según la frecuencia
if frecuencia_seleccionada == "Diaria":
  intervalo_yf = "1d"
  factor_anual = 252
  texto_frecuencia = "diarios (252 ruedas anuales)"
elif frecuencia_seleccionada == "Semanal":
  intervalo_yf = "1wk"
  factor_anual = 52
  texto_frecuencia = "semanales (52 semanas anuales)"
else:
  intervalo_yf = "1mo"
  factor_anual = 12
  texto_frecuencia = "mensuales (12 meses anuales)"

# Rango de fechas
col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
  fecha_inicio = st.date_input("Fecha de Inicio", datetime.date(2023, 1, 1))
with col_f2:
  fecha_fin = st.date_input("Fecha de Fin", datetime.date.today())

# Tasa libre de riesgo estimada para el costo de capital / CAPM
tasa_libre_riesgo = (
    st.sidebar.number_input(
        "Tasa Libre de Riesgo Anual (%)", value=4.0, step=0.25
    )
    / 100
)


# Función para descargar y procesar datos con el intervalo seleccionado
@st.cache_data
def cargar_datos_frecuencia(acc, mkt, inicio, fin, intervalo):
  try:
    tickers_a_descargar = list(set([acc, mkt]))
    datos = yf.download(
        tickers_a_descargar,
        start=inicio,
        end=fin,
        interval=intervalo,
        progress=False,
    )["Close"]
    if isinstance(datos, pd.Series):
      return None, "No se pudieron descargar suficientes datos para los símbolos."
    datos = datos.dropna()
    return datos, None
  except Exception as e:
    return None, str(e)


# Ejecutar descarga
datos_crudos, error = cargar_datos_frecuencia(
    ticker_accion, ticker_mercado, fecha_inicio, fecha_fin, intervalo_yf
)

if error or datos_crudos is None or datos_crudos.empty:
  st.error(
      f"Ocurrió un error al obtener los datos. Verifica los símbolos"
      f" ingresados ({ticker_accion}, {ticker_mercado}). Detalle: {error}"
  )
else:
  # Validar que no se haya seleccionado el mismo activo en ambos campos
  if ticker_accion == ticker_mercado:
    st.warning(
        f"⚠️ **Atención:** Has seleccionado el mismo símbolo (**{ticker_accion}**)"
        " tanto para la acción como para el benchmark. Por favor, ingresa un"
        " activo diferente."
    )
  else:
    if (
        ticker_accion in datos_crudos.columns
        and ticker_mercado in datos_crudos.columns
    ):
      df = datos_crudos[[ticker_accion, ticker_mercado]].copy()
      df.columns = ["Accion", "Mercado"]

      # Calcular retornos porcentuales según la frecuencia elegida
      df["Ret_Accion"] = df["Accion"].pct_change()
      df["Ret_Mercado"] = df["Mercado"].pct_change()
      df = df.dropna()

      # --- CÁLCULOS ESTADÍSTICOS Y FINANCIEROS DINÁMICOS ---
      ret_esp_accion_anual = (1 + df["Ret_Accion"].mean()) ** factor_anual - 1
      ret_esp_mercado_anual = (
          1 + df["Ret_Mercado"].mean()
      ) ** factor_anual - 1

      std_accion_diario = df["Ret_Accion"].std()
      std_accion_anual = std_accion_diario * np.sqrt(factor_anual)
      var_accion_anual = df["Ret_Accion"].var() * factor_anual

      cov_mat = np.cov(df["Ret_Accion"], df["Ret_Mercado"])
      beta = cov_mat[0, 1] / cov_mat[1, 1]

      costo_capital_capm = tasa_libre_riesgo + beta * (
          ret_esp_mercado_anual - tasa_libre_riesgo
      )

      # --- VISUALIZACIÓN EN PANTALLA (RESUMEN EJECUTIVO) ---
      st.markdown(
          "### 📈 Resumen Ejecutivo de Desempeño y Riesgo (Anualizado)"
      )
      st.markdown(f"*(Valores calculados con retornos {texto_frecuencia})*")

      kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

      with kpi1:
        st.metric(
            label=f"Ret. Esp. Anual ({ticker_accion})",
            value=f"{ret_esp_accion_anual:.2%}",
        )
      with kpi2:
        st.metric(
            label=f"Ret. Esp. Anual ({ticker_mercado})",
            value=f"{ret_esp_mercado_anual:.2%}",
        )
      with kpi3:
        st.metric(
            label=f"Varianza σ² ({ticker_accion})",
            value=f"{var_accion_anual:.4f}",
        )
      with kpi4:
        st.metric(
            label=f"Desvío Anual σ ({ticker_accion})",
            value=f"{std_accion_anual:.2%}",
        )
      with kpi5:
        st.metric(label="Beta (β)", value=f"{beta:.2f}")

      st.markdown("---")

      # --- PESTAÑAS DE ANÁLISIS ---
      tab1, tab2, tab3, tab4, tab5 = st.tabs([
          f"📊 Rentabilidad y Distribución ({frecuencia_seleccionada})",
          "📐 Análisis de Regresión y Beta",
          "📋 Datos Históricos",
          "🧮 Cálculos Detallados",
          "💰 Costo de Capital (CAPM)",
      ])

      with tab1:
        st.subheader(
            f"Evolución y Distribución de Retornos {frecuencia_seleccionada}s"
            f" de {ticker_accion}"
        )

        col_g1, col_g2 = st.columns(2)

        with col_g1:
          st.markdown(
              f"**Rentabilidad {frecuencia_seleccionada} — {ticker_accion}**"
          )
          fig_ret = px.line(
              df,
              y="Ret_Accion",
              labels={
                  "index": "Fecha",
                  "Ret_Accion": f"Rentabilidad {frecuencia_seleccionada}",
              },
          )
          fig_ret.update_layout(hovermode="x unified", showlegend=False)
          st.plotly_chart(fig_ret, use_container_width=True)

        with col_g2:
          st.markdown(
              f"**Histograma de Frecuencia — {ticker_accion}**"
          )
          fig_hist = px.histogram(
              df,
              x="Ret_Accion",
              nbins=50,
              labels={
                  "Ret_Accion": f"Rentabilidad {frecuencia_seleccionada}",
                  "count": "Frecuencia de Ocurrencia",
              },
          )
          st.plotly_chart(fig_hist, use_container_width=True)

      with tab2:
        st.subheader(
            "Dispersión de Retornos y Línea de Regresión (Beta)"
        )
        st.markdown(
            "El coeficiente Beta (β) mide la sensibilidad de la acción respecto"
            " al mercado utilizando los datos de frecuencia"
            f" {frecuencia_seleccionada.lower()}."
        )

        m, b_int = np.polyfit(df["Ret_Mercado"], df["Ret_Accion"], 1)
        df["Tendencia_Regresion"] = m * df["Ret_Mercado"] + b_int

        st.info(
            f"**Ecuación de Regresión Estimada:**  \n"
            f"`Retorno_{ticker_accion}` = **{m:.4f}** ×"
            f" `Retorno_{ticker_mercado}` + **{b_int:.6f}**  \n"
            f"*(Donde la pendiente **{m:.2f}** representa el coeficiente Beta)*"
        )

        fig_scatter = px.scatter(
            df,
            x="Ret_Mercado",
            y="Ret_Accion",
            opacity=0.5,
            labels={
                "Ret_Mercado": (
                    f"Retorno {frecuencia_seleccionada} Mercado"
                    f" ({ticker_mercado})"
                ),
                "Ret_Accion": (
                    f"Retorno {frecuencia_seleccionada} Acción"
                    f" ({ticker_accion})"
                ),
            },
        )

        fig_scatter.add_traces(
            go.Scatter(
                x=df["Ret_Mercado"],
                y=df["Tendencia_Regresion"],
                mode="lines",
                name=f"Línea de Regresión (Beta = {beta:.2f})",
                line=dict(color="red", width=2),
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

      with tab3:
        st.subheader("Tabla de Datos Históricos Procesados")
        df_display = df[["Accion", "Mercado", "Ret_Accion", "Ret_Mercado"]].copy()
        df_display.columns = [
            f"Precio {ticker_accion}",
            f"Precio {ticker_mercado}",
            f"Retorno {frecuencia_seleccionada} {ticker_accion}",
            f"Retorno {frecuencia_seleccionada} {ticker_mercado}",
        ]
        st.dataframe(df_display.tail(100), use_container_width=True)

      with tab4:
        st.subheader("🧮 Memoria de Cálculo Detallada y Fórmulas")
        st.markdown(
            "A continuación se detallan las fórmulas matemáticas adaptadas a la"
            f" frecuencia seleccionada (**{frecuencia_seleccionada}**, con factor"
            f" de anualización $N = {factor_anual}$):"
        )

        st.markdown("---")
        st.markdown(f"### 1. Retorno Esperado Anualizado ({ticker_accion})")
        st.markdown(
            "Se calcula componiendo el promedio aritmético de los retornos del"
            f" período a lo largo de los $N = {factor_anual}$ períodos anuales:"
        )
        st.latex(
            r"E(R) = (1 + \bar{R}_{periodo})^{N} - 1 \quad \text{donde } N = "
            + str(factor_anual)
        )

        st.markdown("---")
        st.markdown(f"### 2. Varianza Anualizada ($\sigma^2$)")
        st.markdown(
            "La varianza anual se obtiene escalando la varianza del período"
            f" por el factor anual $N = {factor_anual}$:"
        )
        st.latex(
            r"\sigma^2_{anual} = \sigma^2_{periodo} \times N \quad \text{donde }"
            r" N = " + str(factor_anual)
        )

        st.markdown("---")
        st.markdown(f"### 3. Desvío Estándar Anualizado ($\sigma$)")
        st.markdown(
            "Representa la volatilidad anualizada, obtenida multiplicando el"
            f" desvío del período por la raíz cuadrada de $N = {factor_anual}$:"
        )
        st.latex(
            r"\sigma_{anual} = \sigma_{periodo} \times \sqrt{N} \quad"
            r" \text{donde } N = " + str(factor_anual)
        )

        st.markdown("---")
        st.markdown("### 4. Coeficiente Beta (β) del Activo")
        st.markdown(
            "Mide el riesgo sistemático o sensibilidad del activo frente al"
            " mercado, calculado como la covarianza sobre la varianza de los"
            " retornos del mercado en la frecuencia seleccionada:"
        )
        st.latex(r"\beta = \frac{Cov(R_a, R_m)}{Var(R_m)}")

      with tab5:
        st.subheader("💰 Determinación del Costo de Capital (Modelo CAPM)")
        st.markdown(
            "El modelo de valoración de activos financieros (*Capital Asset"
            " Pricing Model*) calcula el costo de oportunidad del accionista en"
            " función del riesgo sistemático utilizando las tasas anualizadas"
            f" basadas en frecuencia {frecuencia_seleccionada.lower()}:"
        )

        st.latex(r"E(R_i) = R_f + \beta_i \times [E(R_m) - R_f]")

        st.markdown("### Componentes del Modelo (Anualizados):")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
          st.metric(
              label="Tasa Libre de Riesgo ($R_f$)",
              value=f"{tasa_libre_riesgo:.2%}",
          )
          st.metric(
              label=(
                  "Retorno Esperado Anual del Mercado $E(R_m)$"
                  f" ({ticker_mercado})"
              ),
              value=f"{ret_esp_mercado_anual:.2%}",
          )
          prima_mercado = ret_esp_mercado_anual - tasa_libre_riesgo
          st.metric(
              label="Prima de Riesgo de Mercado $[E(R_m) - R_f]$",
              value=f"{prima_mercado:.2%}",
          )

        with col_c2:
          st.metric(label=f"Coeficiente Beta (βᵢ)", value=f"{beta:.2f}")
          prima_activo = beta * prima_mercado
          st.metric(
              label=f"Prima por Riesgo del Activo (β × Prima)",
              value=f"{prima_activo:.2%}",
          )
          st.metric(
              label=f"Costo de Capital Final $E(R_i)$ ({ticker_accion})",
              value=f"{costo_capital_capm:.2%}",
          )
