import streamlit as st
import pandas as pd
from .models.classifier import RiskClassifier
import altair as alt


@st.cache_resource(show_spinner=False)
def load_classifier():
    return RiskClassifier()

def main():
    st.title("💥 Novo Desastre")
    classifier = load_classifier()

    st.markdown(
        """
        Este aplicativo realiza a predição de Alto Impacto de desastres naturais com base em dois fatores: **Disaster Subtype** (Tipo de Desastre) e **Subregion** (Sub-região Continental). **O modelo prevê "Alto Impacto"** quando a probabilidade estimada de o evento afetar mais de 1000 pessoas é maior que 0.5. Use os seletores abaixo para obter uma predição online ou envie um arquivo CSV para predição em lote.
        """
    )

    st.subheader("⚡ Predição Online")

    disaster_translation = {
        "Flood (General)": "Inundação (Geral)",
        "Riverine flood": "Inundação fluvial",
        "Flash flood": "Enchente súbita",
        "Heat wave": "Onda de calor",
        "Tropical cyclone": "Ciclone tropical",
        "Blizzard/Winter storm": "Nevasca/Tempestade de inverno",
        "Landslide (wet)": "Deslizamento de terra (úmido)",
        "Viral disease": "Doença viral",
        "Cold wave": "Onda de frio",
        "Ground movement": "Movimento do solo",
        "Tornado": "Tornado",
        "Severe weather": "Clima severo",
        "Drought": "Seca",
        "Ash fall": "Queda de cinzas",
        "Lightning/Thunderstorms": "Relâmpagos/Trovoadas",
        "Bacterial disease": "Doença bacteriana",
        "Forest fire": "Incêndio florestal",
        "Extra-tropical storm": "Tempestade extratropical",
        "Storm (General)": "Tempestade (Geral)"
    }

    subregion_translation = {
        "Sub-Saharan Africa": "África Subsaariana",
        "South-eastern Asia": "Sudeste Asiático",
        "Eastern Europe": "Europa Oriental",
        "Western Asia": "Ásia Ocidental",
        "Central Asia": "Ásia Central",
        "Latin America and the Caribbean": "América Latina e Caribe",
        "Southern Europe": "Europa Meridional",
        "Northern America": "América do Norte",
        "Western Europe": "Europa Ocidental",
        "Southern Asia": "Sul da Ásia",
        "Eastern Asia": "Leste Asiático",
        "Northern Europe": "Europa Setentrional",
        "Northern Africa": "Norte da África",
        "Micronesia": "Micronésia",
        "Australia and New Zealand": "Austrália e Nova Zelândia",
        "Polynesia": "Polinésia",
        "Melanesia": "Melanésia"
    }

    # Listas de opções com tradução
    disaster_options = [""] + list(disaster_translation.values())
    subregion_options = [""] + list(subregion_translation.values())

    # Layout de colunas para seleção online
    col1, col2 = st.columns(2)
    with col1:
        selected_disaster_pt = st.selectbox("Tipo de Desastre", disaster_options, index=0)
    with col2:
        selected_subregion_pt = st.selectbox("Sub-região Continental", subregion_options, index=0)

    reverse_disaster = {v: k for k, v in disaster_translation.items()}
    reverse_subregion = {v: k for k, v in subregion_translation.items()}

    selected_disaster = reverse_disaster.get(selected_disaster_pt, "")
    selected_subregion = reverse_subregion.get(selected_subregion_pt, "")

    # Predição online: apenas se ambos selecionados
    if selected_disaster and selected_subregion:
        sample = {
            "Disaster Subtype": selected_disaster,
            "Subregion": selected_subregion
        }
        result = classifier.predict_online(sample)
        prob_0, prob_1 = result["probability_score"]
        label = int(result["predicted_label"])

        if label == 0:
            st.markdown(f"""
                <div style="background-color: #DCECF9; color: #2A4D8F; padding: 20px; border-radius: 5px; font-size: 18px; font-weight: bold; text-align: center;">
                    🛡️ Desastre de Impacto Moderado 🛡️
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #FCE4E4; color: #7A2B2B; padding: 20px; border-radius: 5px; font-size: 18px; font-weight: bold; text-align: center;">
                    🚨 Desastre de Alto Impacto 🚨
                </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f"**Para o tipo** `{selected_disaster_pt}` **e** `{selected_subregion_pt}`, **a probabilidade de Alto Impacto é** `{prob_1:.3f}`."
        )
        

    st.markdown("---")

    # Seção de predição em lote
    st.subheader("📦 Predição em Lote")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Envie o CSV para predição em lote (formato conforme Exemplo)", type=["csv"])
    with col2:
        st.write("")
        st.write("")
        st.write("")
        
        file_path = "FrontEndandMobileDevelopment/data/processed/app_test.csv"
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Baixar Exemplo CSV",
                data=f,
                file_name="app_test.csv",
                mime="text/csv"
            )


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin1')

        # Realiza predição em lote
        batch_result = classifier.predict_batch(df)

        # Renomeia colunas para nomes amigáveis
        batch_result = batch_result.rename(columns={
            "Disaster Subtype": "Tipo de Desastre",
            "Subregion": "Sub-região",
            "predicted_label": "Alto Impacto",
            "probability_class_0": "Probabilidade Não",
            "probability_class_1": "Probabilidade Sim"
        })

        # Cria coluna com labels legíveis para o impacto
        impacto_map = {1.0: 'Alto', 0.0: 'Moderado'}
        batch_result['Nível de Impacto'] = batch_result['Alto Impacto'].map(impacto_map)

        # Exibe resultados
        st.markdown("**Resultados da predição em lote:**")
        st.dataframe(batch_result, height=300, hide_index=True)

        # Análise descritiva
        st.subheader("📊 Análise Descritiva")
        st.markdown("Esta seção apresenta como as categorias se comportam em relação ao rótulo previsto.")

        cores = {'Alto': "#d15151", 'Moderado': "#607dd3"}

        col1, col2 = st.columns(2)

        with col1:
            categoria = st.radio("Selecione a categoria para visualização:", ['Tipo de Desastre', 'Sub-região'], horizontal=True)

        with col2:
            tipo_grafico = st.radio("Tipo de gráfico:", ['Contagem', 'Percentual'], horizontal=True)

        # Preparar dados base
        df_long = batch_result[[categoria, 'Nível de Impacto']].copy()
        df_long['count'] = 1
        df_grouped = df_long.groupby([categoria, 'Nível de Impacto']).count().reset_index()

        if tipo_grafico == 'Contagem':
            chart = alt.Chart(df_grouped).mark_bar().encode(
                y=alt.Y(f"{categoria}:N", title=categoria),
                x=alt.X('count:Q', stack='zero', title='Contagem'),
                color=alt.Color(
                    'Nível de Impacto:N',
                    scale=alt.Scale(domain=['Alto', 'Moderado'], range=[cores['Alto'], cores['Moderado']]),
                    legend=alt.Legend(title='Nível de Impacto', orient='bottom')
                ),
                tooltip=[categoria, 'Nível de Impacto', 'count']
            ).properties(
                width=700,
                height=700,
                title=f'Distribuição de Nível de Impacto por {categoria}'
            ).interactive()

        else:  # Percentual
            total_por_cat = df_grouped.groupby(categoria)['count'].transform('sum')
            df_grouped['percent'] = df_grouped['count'] / total_por_cat * 100

            chart = alt.Chart(df_grouped).mark_bar().encode(
                y=alt.Y(f"{categoria}:N", title=categoria),
                x=alt.X('percent:Q', title='Percentual (%)'),
                color=alt.Color(
                    'Nível de Impacto:N',
                    scale=alt.Scale(domain=['Alto', 'Moderado'], range=[cores['Alto'], cores['Moderado']]),
                    legend=alt.Legend(title='Nível de Impacto', orient='bottom')
                ),
                tooltip=[categoria, 'Nível de Impacto', alt.Tooltip('percent:Q', format='.1f')]
            ).properties(
                width=700,
                height=700,
                title=f'Proporção percentual de Nível de Impacto por {categoria}'
            ).interactive()

        st.altair_chart(chart)