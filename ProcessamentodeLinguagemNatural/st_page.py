import streamlit as st
import pandas as pd
from .src.pipeline import OnlineDisasterMessagePipeline

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return OnlineDisasterMessagePipeline()

def main():
    pipeline = load_pipeline()
    st.title("📢 Mensagem Urgente")
    st.markdown("**Digite sua mensagem de socorro.** Atenção: este classificador tem a limitação de ter sido treinado exclusivamente em mensagens sintéticas de socorro. "
                "Mensagens fora desse contexto, como **trotes ou textos genéricos, gerarão falsos positivos** na maior parte dos casos. "
                "Isso foi apenas minimamente mitigado com o uso de threshold de 0.6 de confiança mínima.")

    # Inicializa ou recupera o histórico de predições
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Entrada do usuário
    text_input = st.text_area("✉️ Mensagem", height=100)

    if st.button("🔍 Classificar"):
        if text_input.strip() == "":
            st.warning("Por favor, insira uma mensagem.")
        else:
            result = pipeline.predict(text_input)

            st.subheader("📋 Resultado da Classificação")
            st.markdown(f"**Texto:** {result['text']}")

            pred = result['predictions']

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌪️ Tipo de Desastre")
                st.markdown(f"**Predição:** `{pred['Desastre_Predicted']}`")
                st.progress(pred['Desastre_Confidence'])
                st.caption(f"Confiança: {pred['Desastre_Confidence']:.2f}")

            with col2:
                st.markdown("### 🚑 Nível de Urgência")
                st.markdown(f"**Predição:** `{pred['Urgencia_Predicted']}`")
                st.progress(pred['Urgencia_Confidence'])
                st.caption(f"Confiança: {pred['Urgencia_Confidence']:.2f}")
            
            if result.get("entities"):
                with st.expander("🔎 Entidades Reconhecidas"):
                    for ent in result["entities"]:
                        st.write(f"- **{ent['label']}**: _{ent['text']}_ (posição: {ent['start_char']}–{ent['end_char']})")

            # Exibir todas as probabilidades
            with st.expander("📊 Ver todas as probabilidades"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Desastres:**")
                    for k, v in pred.items():
                        if k.startswith("Desastre_") and k not in ["Desastre_Predicted", "Desastre_Confidence"]:
                            st.write(f"{k.replace('Desastre_', '').replace('_', ' ')}: {v:.2f}")
                with col2:
                    st.markdown("**Urgência:**")
                    for k, v in pred.items():
                        if k.startswith("Urgencia_") and k not in ["Urgencia_Predicted", "Urgencia_Confidence", "Urgencia_Score"]:
                            st.write(f"{k.replace('Urgencia_', '').replace('_', ' ')}: {v:.2f}")

            with st.expander("🤖 Resposta do DisasterBot"):
                st.code(result.get("disasterbot_response", "Sem resposta"), language="text")

            # Adiciona resultado à sessão
            st.session_state["history"].append({
                "Mensagem": result["text"],
                "Tipo de Desastre": pred["Desastre_Predicted"],
                "Nível de Urgência": pred["Urgencia_Predicted"],
                "Score Agregado": pred["Urgencia_Score"]
            })

    # Exibir tabela ordenada por score agregado
    if st.session_state["history"]:
        st.subheader("📈 Histórico de Classificações")
        df_history = pd.DataFrame(st.session_state["history"])
        df_sorted = df_history.sort_values(by="Score Agregado", ascending=False)
        st.dataframe(df_sorted)
