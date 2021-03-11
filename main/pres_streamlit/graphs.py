#import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from wordcloud import WordCloud

df = pd.read_csv('data/info_data_job_market_research.csv')


def graph_1(df):
    puesto_ds = df[df['Tipo puesto'] == 'Data Scientist']

    puesto_ds = pd.Series(' '.join(puesto_ds['Puesto ofertado']).lower().split()).value_counts()[
                :30]

    rem_word_puesto = ['-', 'and', '&', 'de', 'in', 'for']

    for i in rem_word_puesto:
        if i in puesto_ds:
            puesto_ds = puesto_ds.drop(i)

    puesto_ds = puesto_ds[:12]

    puesto_ds.rename('Suma', inplace=True)

    df_puesto_ds = pd.DataFrame(puesto_ds)

    df_puesto_ds.reset_index(inplace=True)

    df_puesto_ds.rename(columns={'index': 'Palabras en DA'}, inplace=True)

    fig_graph_1 = px.bar(df_puesto_ds, x='Palabras en DA', y='Suma',
                 labels={'x': 'Palabras en DA', 'y': 'Suma'})

    return fig_graph_1


def graph_2(df):
    diasact_da = df[df['Tipo puesto'] == 'Data Analyst']

    diasact_da = diasact_da['Días activos'].copy()

    lista_dias = ['segundo', 'segundos', 'minuto', 'minutos', 'hora', 'horas', 'día', 'días']

    for i in lista_dias:
        diasact_da[diasact_da.str.contains(i)] = '03-08-21'

    diasact_da[diasact_da.str.contains(' 1 semana')] = '03-01-21'
    diasact_da[diasact_da.str.contains(' 2 semanas')] = '02-22-21'
    diasact_da[diasact_da.str.contains(' 3 semanas')] = '02-15-21'
    diasact_da[diasact_da.str.contains(' 4 semanas')] = '02-08-21'
    diasact_da[diasact_da.str.contains(' 1 mes')] = '02-01-21'
    diasact_da[diasact_da.str.contains(' 2 meses')] = '01-25-21'
    diasact_da[diasact_da.str.contains(' 3 meses')] = '01-18-21'
    diasact_da[diasact_da.str.contains(' 4 meses')] = '01-11-21'
    diasact_da[diasact_da.str.contains(' 5 meses')] = '01-04-21'

    diasact_da = diasact_da.value_counts()

    diasact_da.drop('no_data', inplace=True)

    df_diasact_da = pd.DataFrame(diasact_da)

    df_diasact_da.reset_index(inplace=True)

    df_diasact_da.rename(columns={'index': 'Fechas'}, inplace=True)

    df_diasact_da['Fechas'] = pd.to_datetime(df_diasact_da['Fechas'])

    df_diasact_da.sort_values('Fechas', ascending=True, inplace=True)

    fig_graph_2 = px.bar(df_diasact_da, x='Fechas', y='Días activos',
                 labels={'x': 'Palabras en DA', 'y': 'Suma'})
    fig_graph_2['layout']['xaxis']['autorange'] = "reversed"
    return fig_graph_2