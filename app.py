from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Tạo Flask app
app = Flask(__name__)

# Tải mô hình đã lưu
model = load_model('model_CNN.h5')

# Hàm dự đoán từ hình ảnh
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(512, 512))  # Điều chỉnh kích thước theo yêu cầu của mô hình
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    img_array /= 255.  # Chuẩn hóa dữ liệu

    predictions = model.predict(img_array)
    return predictions

# Trang chính
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Giao diện web

# Route xử lý upload và dự đoán
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"})
    
    file = request.files['file']
    
    if file:
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)
        
        # Dự đoán kết quả từ hình ảnh
        predictions = model_predict(file_path, model)
        
        # Xử lý kết quả dự đoán (ví dụ: trả về nhãn có xác suất cao nhất)
        result = np.argmax(predictions, axis=1)
        return jsonify({"prediction": str(result)})

if __name__ == '__main__':
    app.run(debug=False)
