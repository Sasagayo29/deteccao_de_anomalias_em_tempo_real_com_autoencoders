import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
import time

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Detecção de Anomalias",
    page_icon="📊",
    layout="wide"
)

# --- Constantes ---
TIME_STEPS = 20  # O mesmo valor da Célula 2

# --- Carregamento dos Artefatos (com cache) ---


@st.cache_resource
def load_model_and_scaler():
    """Carrega o modelo e o scaler salvos."""
    try:
        # Caminhos
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent
        MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder_model.h5"
        SCALER_PATH = PROJECT_ROOT / "models" / "anomaly_scaler.pkl"

        print(f"Carregando modelo de: {MODEL_PATH}")
        # CORREÇÃO 1: Adicionado compile=False
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print(f"Carregando scaler de: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)

        print("Modelo e scaler carregados com sucesso.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Erro: Arquivos de modelo ou scaler não encontrados.")
        st.error(
            f"Verifique se 'autoencoder_model.h5' e 'anomaly_scaler.pkl' estão na pasta 'models/'.")
        return None, None
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {e}")
        return None, None


model, scaler = load_model_and_scaler()


@st.cache_data
# Passar o scaler para o cache funcionar
def load_simulation_data(_scaler_obj):
    """Carrega um arquivo de teste para simular o streaming."""
    try:
        # Caminhos
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent
        DATA_ROOT = PROJECT_ROOT / "data" / "raw"

        DATA_PATH = DATA_ROOT / "1.csv"

        df = pd.read_csv(DATA_PATH, sep=';', parse_dates=[
                         'datetime'], index_col='datetime')

        sensor_cols = [col for col in df.columns if col not in [
            'anomaly', 'changepoint']]

        # Normalizar os dados
        df_scaled_values = _scaler_obj.transform(df[sensor_cols])

        true_anomalies = df['anomaly'].values

        return df_scaled_values, true_anomalies, df.index, len(sensor_cols)

    except FileNotFoundError:
        st.error(
            f"Erro: Arquivo de dados '1.csv' não foi encontrado em '{DATA_PATH}'.")
        st.error("Verifique se o arquivo '1.csv' está na sua pasta 'data/raw/'.")
        return None, None, None, 0
    except Exception as e:
        st.error(f"Erro ao carregar dados de simulação: {e}")
        return None, None, None, 0


# --- Interface do Usuário (UI) ---
st.title("🛰️ Dashboard de Detecção de Anomalias em Tempo Real")
st.markdown("Este dashboard simula um streaming de dados de sensores e usa um Autoencoder (LSTM) para detectar anomalias.")

if model is not None and scaler is not None:
    # Passamos o 'scaler' para a função em cache
    df_scaled_values, true_anomalies, data_index, n_features = load_simulation_data(
        scaler)

    if df_scaled_values is not None:

        st.sidebar.header("Configurações da Simulação")

        st.sidebar.warning("👇 Cole o seu 'Limiar (μ + 3σ)' da Célula 3 aqui!")
        threshold = st.sidebar.number_input(
            "Limiar de Anomalia (Threshold)",
            min_value=0.0,
            value=0.05,  # Valor padrão, use o seu
            step=0.001,
            format="%.6f"
        )

        st.sidebar.markdown("---")
        start_button = st.sidebar.button("Iniciar Simulação 🚀")
        st.sidebar.markdown("---")
        st.sidebar.info(f"Modelo carregado.\n\n"
                        f"Sequência: {TIME_STEPS} passos\n\n"
                        f"Features: {n_features} sensores")

        # Placeholders do dashboard
        col1, col2 = st.columns(2)
        with col1:
            status_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()

        chart_placeholder = st.empty()

        if start_button:
            st.success("Iniciando simulação...")

            error_history = []

            # Loop de Simulação
            for i in range(len(df_scaled_values)):

                current_window = df_scaled_values[max(
                    0, i - TIME_STEPS + 1): i + 1]

                padded_window = np.zeros((TIME_STEPS, n_features))
                padded_window[-current_window.shape[0]:] = current_window

                model_input = np.reshape(
                    padded_window, (1, TIME_STEPS, n_features))

                reconstruction = model.predict(model_input, verbose=0)
                error = np.mean(
                    np.abs(reconstruction - model_input), axis=(1, 2))[0]

                is_anomaly = error > threshold
                true_status = true_anomalies[i]

                # Atualizar UI
                with status_placeholder.container():
                    if is_anomaly:
                        st.error(f"🚨 ANOMALIA DETECTADA! (Erro: {error:.6f})")
                    else:
                        st.success(f"✅ Operação Normal (Erro: {error:.6f})")

                with metrics_placeholder.container():
                    st.metric(label="Erro de Reconstrução",
                              value=f"{error:.6f}")
                    st.metric(label="Status Real (do Dataset)",
                              value="ANOMALIA" if true_status == 1 else "Normal")

                # Atualizar Gráfico
                error_history.append({
                    'Timestamp': data_index[i],
                    'Erro': error,
                    'Limiar': threshold
                })

                df_plot = pd.DataFrame(error_history).set_index('Timestamp')
                # Plotar só os últimos 200
                chart_placeholder.line_chart(df_plot.tail(200))

                time.sleep(0.05)

            st.success("Simulação concluída!")
