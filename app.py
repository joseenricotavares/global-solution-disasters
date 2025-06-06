import streamlit as st
from streamlit_option_menu import option_menu
import home
from ProcessamentodeLinguagemNatural import st_page as pln
from FrontEndandMobileDevelopment import st_page as femd

st.set_page_config(
    page_title="Global Solution",
    page_icon="🌍",
    layout="centered"
)

with st.sidebar:
    pagina = option_menu(
        "Menu", 
        ["Home", "💥 Novo Desastre", "📢 Mensagem Urgente"], 
        menu_icon="cast", 
        default_index=0
    )

if pagina == "Home":
    home.main()  
elif pagina == "💥 Novo Desastre":
    femd.main()
elif pagina == "📢 Mensagem Urgente":
    pln.main()