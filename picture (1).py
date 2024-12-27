import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# 모델 로드
model = load_model('monalisa_model3.h5')

def preprocess_image(image):
    image = image.resize((150, 150))  # 모델에 맞는 이미지 크기로 조정
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    confidence = float(prediction[0])  # 예측 확률
    if 1-confidence > 0.5:
        return "모나리자", 1-confidence
    else:
        return "모나리자가 아님", 1-confidence

# Streamlit 애플리케이션 설정
st.title("모나리자 이미지 분류기")
st.write("Keras CNN 모델을 사용하여 모나리자인지 여부를 예측합니다.")

# 이미지 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드 하세요.", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.write("")
    st.write("이미지 분류 중...")

    # 예측
    result, confidence = predict(image)
    st.write(f"예측 결과: {result}")
    st.write(f"예측 확률: {confidence * 100:.2f}%")
