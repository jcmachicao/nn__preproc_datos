# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import base64

c1, c2, c3 = st.beta_columns([2,2,2])
with c3:
    st.image('gdmk.png', width=150, caption='www.gestiodinamica.com')

st.subheader('COLECCIÓN: MODELAMIENTO DE REDES NEURONALES')
st.title('Preparación de Data')

st.write('Este aplicativo procesa la base de datos de prueba para predecir el \
         valor con el modelo previsto. \
         Subir los datos según el formato previsto.')

st.subheader('Carga de Datos')

st.write('*Instrucciones*: El archivo a cargar puede tener cualquier nombre con \
         la extensión **___.xlsx** y debe contener dos hojas.\
         Una primera llamada **data_orig** y una segunda llamada **data_futura**. \
         Ambas deben tener la misma estructura de columnas y la variable de \
         salida debe denominarse como **out** (en el encabezado de columna).')

def prep_data_nn(data, data_fut):
    
    data['futuro'] = False
    data_fut['futuro'] = True
    data_tot = pd.concat([data, data_fut], axis=0)
    data_tot = data_tot.drop(['out'], axis=1)
    
    cats, nums = [], []
    for col, dt in zip(data_tot.columns, data_tot.dtypes):
        if dt == 'object': cats.append(col) 
        else: nums.append(col)
    
    unicos = []
    for col in cats:
        unicos.append(list(data_tot[col].unique()))
    
    data_tot_wide_cats = pd.get_dummies(data_tot[cats], drop_first=True)
    data_tot_wide = pd.concat([data_tot[nums], data_tot_wide_cats], axis=1)
    data_norm = data_tot_wide / np.max(data_tot_wide, axis=0)
    data_norm_in = data_norm[data_norm.futuro == 0]
    data_norm_in = data_norm_in.drop(['futuro'], axis=1)
    X_in_cols = data_norm_in.columns
    data_y = pd.factorize(data.out)
    y_cats = data_y[1]
    X_in = np.array(data_norm_in, dtype='float64')
    y_in = np.array(data_y[0], dtype='float32').reshape(-1,1)
   
    return X_in, y_in, X_in_cols, y_cats, unicos, cats, nums

file = st.file_uploader('Seleccione un archivo')

if file is not None: 
    
    @st.cache(allow_output_mutation=True)
    def load(file):
        data = pd.read_excel(file, sheet_name='data_orig')
        data_fut = pd.read_excel(file, sheet_name='data_futura')
        return data, data_fut
    
    data, data_fut = load(file)
        
    st.write('Filas, Columnas de Data de Entrenamiento: ', data.shape)
    st.write('Filas, Columnas de Data Potencial Futura: ', data_fut.shape)
    
    st.subheader('Datos de Entrenamiento')

    X_in, y_in, X_in_cols, y_cats, unicos, cats, nums = prep_data_nn(data, data_fut)

    st.write(tuple(cats))
    option = st.selectbox('Opción: ', tuple(cats))
 
    st.bar_chart(data[option].value_counts(), width=500, use_container_width=False)
    
    st.write('Geometría de Data X: ', X_in.shape)
    st.write('Geometría de Data y: ', y_in.shape)
    st.write('Valores únicos de variables categóricas: ', unicos)
    st.write('Columnas identificadas para X con dummies: ', X_in_cols)
    st.write('Categorías identificadas para y: ', y_cats)
    
    st.subheader('Confirmación')
    
    st.write('Con estos archivos ya es posible entrenar la red neuronal. Los archivos pueden descargarse abajo. Muchas gracias.')
    
    df = pd.DataFrame(X_in)
    df.columns = list(X_in_cols)
    df['out'] = y_in
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data_in_modelo.csv">Descargar Archivo</a>'
    
    st.write('Archivo X_in: ')
    st.markdown(href, unsafe_allow_html=True)
