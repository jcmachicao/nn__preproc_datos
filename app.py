# -*- coding: utf-8 -*-

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import base64

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

c1, c2, c3 = st.beta_columns([2,2,2])
with c3:
    st.image('gdmk.png', width=150, caption='www.gestiodinamica.com')

st.subheader('COLECCIÓN: MODELAMIENTO DE REDES NEURONALES')
st.title('Predicción con Modelo Existente')

st.write('Este aplicativo procesa la base de datos de prueba para predecir el \
         valor con el modelo previsto. Subir los datos según el formato previsto.')

st.subheader('Carga de Datos')

st.write('*Instrucciones*: El archivo a cargar puede tener cualquier nombre con \
        la extensión **___.xlsx** y debe contener la data a predecir en el modelo\
        en una hojas llamada **data_testing** con la misma estructura de columnas\
        y las mismas categorías usadas en la data original. La estructura original \
        de columnas se muestra abajo.' 
)

# FUNCIONES Y CLASES

def prep_data_nn(data, cols):
    
    data.columns = cols

    cats, nums = [], []
    for col, dt in zip(data.columns, data.dtypes):
        if dt == 'object': cats.append(col) 
        else: nums.append(col)
    
    unicos = []
    for col in cats:
        unicos.append(list(data[col].unique()))
    
    data_wide_cats = pd.get_dummies(data[cats], drop_first=True)
    cats_dums = data_wide_cats.columns
    data_tot = pd.concat([data[nums], data_wide_cats], axis=1)
     
    return data_tot, unicos, cats_dums

# Arquitectura

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_cols, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 32)
    self.fc4 = nn.Linear(32, n_cats)
    self.dp1 = nn.Dropout(0.15)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dp1(F.relu(self.fc2(x)))
    x = self.dp1(F.relu(self.fc3(x)))
    x = self.fc4(x)

    return F.log_softmax(x, dim=1)

# OPERACIONES

file = st.file_uploader('Seleccione un archivo: ')

if file is not None: 
    
    dataxb = pd.read_excel(file, sheet_name='test_x')
    st.write('Filas, Columnas de Data de Entrenamiento: ', dataxb.shape)
    
    st.subheader('Conversión para Entrenamiento')
    
    colsb = list(dataxb.columns)
    
    #data_X, unicos, cats_dums = prep_data_nn(dataxb, colsb)

    st.write('Geometría de Data: ', dataxb.shape)
    #st.write(tuple(cats_dums))

    st.subheader('Visualización')

    option_a = st.selectbox('Columna Filtro: ', tuple(dataxb.columns))
    option_b = st.selectbox('Categorías: ', tuple(dataxb[option_a].unique()))

    df3 = dataxb[dataxb[option_a]==option_b]
    fig = px.parallel_coordinates(df3)
    fig.update_layout(width=800)
    st.write(fig)

    st.write('Columnas identificadas para X con dummies: ', len(dataxb.columns))
    st.write('... Confirmado, listo para testing.')
    st.write('Archivo data lista para testing: ')
    st.markdown(href, unsafe_allow_html=True)

    st.subheader('Generación de Predicciones con Data')
    mms = MinMaxScaler()
    data_np_01 = mms.fit_transform(data_np)
    X_ts = data_np_01
    st.write(X_ts.shape)
    
    # Arquitectura
    model = NetClas()
    nom_model = 'modelo_RET01.pth'
    modelo_load = nom_model
    model.load_state_dict(torch.load(modelo_load, map_location=torch.device('cpu')))
    output = model(torch.tensor(X_ts).float())
    y_hat = torch.argmax(output.detach(), 1).numpy()
    st.write(y_hat)
    
    dataxa = dataxb
    dataxa['y_hat'] = y_hat
    
    st.subheader('Visualización de Predicciones')
    
    op_pred = st.selectbox('Positivas o Negativas: ', tuple(dataxa.y_hat.unique()))
    dataxa2 = dataxa[dataxa.y_hat==op_pred]
    st.write(dataxa2.shape)
    st.write('Ratio Resultado Positivo: ', round(len(dataxa2)/len(dataxa),2))
    
    df4 = dataxa2
    fig = px.parallel_coordinates(df4)
    fig.update_layout(width=800)
    st.write(fig)
    
    csv = dataxa2.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data_predicciones.csv">Descargar Archivo</a>'
    
    st.write('Archivo con predicciones: ')
    st.markdown(href, unsafe_allow_html=True)
