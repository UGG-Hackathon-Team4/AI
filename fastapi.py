from fastapi import FastAPI, File, UploadFile
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from PIL import Image as PILImage

app = FastAPI()

# 모델 로드 (compile=False로 경고 제거)
model = load_model("monalisa_model3.h5", compile=False)

# 이미지 전처리 함수
def preprocess_image(image_file):
    image = PILImage.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((150, 150))  # 모델에 맞는 입력 크기로 조정
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    # 업로드된 이미지 읽기
    image_data = await image.read()
    img_array = preprocess_image(image_data)

    # 모델 예측
    prediction = model.predict(img_array)
    result = "모나리자" if prediction[0][0] > 0.5 else "모나리자가 아님"

    return {"prediction": result}

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Jupyter에서 실행 시 필요
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
