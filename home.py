import streamlit as st

def main():
    st.title("🌍 Global Solution - 2025.1")
    st.markdown(
        """
        Este ambiente interativo apresenta duas soluções desenvolvidas no contexto acadêmico da FIAP, voltadas à prevenção e resposta a desastres naturais. Ambas as aplicações demonstram como a tecnologia pode ser uma aliada fundamental na redução de riscos e no suporte à tomada de decisão em situações críticas.
        """
    )

    st.markdown("---")

    st.header("💥 Novo Desastre")
    st.markdown(
        """
        Desenvolvida na disciplina **Front-End and Mobile Development**, esta aplicação realiza **predições sobre o potencial de impacto de desastres naturais** com base no tipo de evento e na sub-região geográfica. Utilizando aprendizado de máquina, o sistema classifica automaticamente o risco como **Alto** ou **Moderado**, auxiliando organizações e autoridades a priorizarem ações de resposta e mitigação. A ferramenta permite uso **interativo (online)** e por **envio de arquivos CSV (em lote)**.
        """
    )

    st.markdown("---")

    st.header("📢 Mensagem Urgente")
    st.markdown(
        """
        Criada na disciplina **Processamento de Linguagem Natural** do curso de **Inteligência Artificial**, esta aplicação permite que mensagens de socorro sejam analisadas automaticamente por um classificador treinado para identificar **textos relacionados a emergências**. O objetivo é apoiar sistemas automatizados de triagem de mensagens em contextos de desastre. Importante: o classificador foi treinado com dados sintéticos e pode apresentar limitações em contextos reais.
        """
    )

    st.markdown("---")
    st.markdown("🧪 *Este aplicativo é um protótipo desenvolvido para fins acadêmicos e experimentais.*")
