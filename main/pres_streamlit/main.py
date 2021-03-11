import streamlit as st
import functions as ft
import pandas as pd
import graphs as gph

ft.config_page()

st.title('Data Analyst & Scientist research')

df_jobs = pd.read_csv(ft.csv_path)


menu = st.sidebar.selectbox('Seleccionar menú:', ('Principal', 'Gráficos', 'Filtros'))
if menu == 'Principal':
    ft.home()

if menu == 'Gráficos':
    with st.beta_expander('Ver mapa de puestos:'):
        st.map(df_jobs)

    with st.beta_expander('Ver gráfico de:'):
        graph_1 = gph.graph_1(gph.df)
        st.plotly_chart(graph_1)

    with st.beta_expander('Ver gráfico de:'):
        graph_2 = gph.graph_2(gph.df)
        st.plotly_chart(graph_2)

if menu == 'Filtros':
    ft.filtros(df_jobs)