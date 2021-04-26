import streamlit as st
import pandas as pd
import cv2

from PIL import Image
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model

def config_page():
    st.set_page_config(page_title='Dog breeds predictions', layout="wide")
config_page()

model = load_model("data/8_3_1_cb_model.h5")

st.title('Dog breeds predictions')
img_header = Image.open('data/dog_header.jpg')
st.image(img_header, use_column_width='auto')

st.markdown('''### ¡Aquí podrás averiguar que tipo de raza es cualquier perro!''')
st.markdown('''##### Solo tienes que cargar la imagen y automáticamente te dirá la raza y una breve descripción.''')

st.write("#")

with st.beta_expander('Pincha aquí si quieres saber más acerca del algoritmo.'):
    st.markdown('''Este predictor se ha hecho mediante Transfer learning, con un dataset de Kaggle.
                El modelo final, está entrenado sobre Xception (Extreme Inception).
                Una variante de la arquitectura GooLeNet
                habiendo procesado 350 millones de imágenes y 17.000 clases.''')
    st.markdown('''Mientras que una capa convolucional corriente usa filtros que intentan capturar 
                simultáneamente patrones espaciales y patrones de canales cruzados 
                (p. Ej., Boca + nariz + ojos = cara), una capa convolucional separable asume 
                que los patrones espaciales y cruzados se pueden modelar por separado.
                Por lo tanto, se compone de dos partes:
                La primera parte aplica un solo filtro espacial para cada mapa de características de entrada,
                luego la segunda parte busca exclusivamente patrones de canales cruzados;
                es solo una capa convolucional regular con filtros 1 × 1.''')

st.write("#")

st.header('Carga tu imagen en JPG:')

buffer = st.file_uploader('', type=['jpg'])
temp_file = NamedTemporaryFile(delete=False)

if buffer:
    temp_file.write(buffer.getvalue())
    img = (cv2.resize(cv2.imread(temp_file.name),(200, 200)))
    img = img/255
    img = img.reshape(1, 200, 200, 3)

st.write('Esta es tu imagen:')
st.image(buffer, use_column_width='auto')

pred = model.predict(img)
pred = pred.argmax()

breeds = pd.read_csv('data/breeds_df.csv')

for i in range(len(breeds)):
    if breeds['id'][i] == pred:
        breed_name = breeds['name_breed'][i]
        desc_breed = breeds['description'][i]

st.write("#")

st.header('La raza es: ' + breed_name)
st.write(desc_breed)

st.write('Ejemplo de la imagen predicha por el algoritmo:')
img_breed = Image.open('dogs_photos/' + pred.astype('str') + '.jpg')
st.image(img_breed, use_column_width='auto')
