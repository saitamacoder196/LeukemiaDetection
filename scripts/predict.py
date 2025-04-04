#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script dự đoán ảnh tế bào.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_image

def load_model(model_name, weights_path):
    """
    Tải mô hình với weights đã huấn luyện.
    
    Args:
        model_name (str): Tên mô hình cần tải.
        weights_path (str): Đường dẫn tới file weights.
        
    Returns:
        model: Mô hình đã tải weights.
    """
    from tensorflow.keras.models import load_model
    
    try:
        model = load_model(weights_path)
        print(f"Đã tải mô hình từ {weights_path}")
        return model
    except:
        print(f"Không thể tải mô hình từ {weights_path}. Tạo mô hình mới và tải weights...")
        
        if model_name == 'resnet':
            from models.resnet import create_model
        elif model_name == 'efficientnet':
            from models.efficientnet import create_model
        elif model_name == 'inception':
            from models.inception import create_model
        elif model_name == 'xception':
            from models.xception import create_model
        elif model_name == 'regnetx':
            from models.regnetx import create_model
        elif model_name == 'convnext':
            from models.convnext import create_model
        elif model_name == 'alexnet':
            from models.alexnet import create_model
        elif model_name == 'vgg':
            from models.vgg import create_model
        elif model_name == 'yolo':
            from models.yolo import create_model
        else:
            raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")
        
        model = create_model()
        model.load_weights(weights_path)
        print(f"Đã tải weights từ {weights_path}")
        
        return model

def predict_image(model_name, weights_path, image_path):
    """
    Dự đoán loại tế bào trong ảnh.
    
    Args:
        model_name (str): Tên mô hình cần sử dụng.
        weights_path (str): Đường dẫn tới file weights.
        image_path (str): Đường dẫn tới ảnh cần dự đoán.
        
    Returns:
        str: Kết quả dự đoán.
    """
    print(f"Đang dự đoán ảnh {image_path} bằng mô hình {model_name}...")
    
    # Tải mô hình
    model = load_model(model_name, weights_path)
    
    # Tiền xử lý ảnh
    img = preprocess_image(image_path)
    
    # Dự đoán
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index] * 100
    
    # Tên các lớp
    class_names = ['Benign', 'Early Pre B', 'Pre B', 'Pro B']
    predicted_class = class_names[class_index]
    
    # Hiển thị kết quả
    result = f"Kết quả dự đoán: {predicted_class} (độ tin cậy: {confidence:.2f}%)"
    print(result)
    
    # Hiển thị ảnh
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img_display)
    plt.title(result)
    plt.axis('off')
    
    # Lưu ảnh kết quả
    results_dir = os.path.join('results', 'predictions')
    os.makedirs(results_dir, exist_ok=True)
    
    image_filename = os.path.basename(image_path)
    save_path = os.path.join(results_dir, f"{os.path.splitext(image_filename)[0]}_{predicted_class}.png")
    plt.savefig(save_path)
    
    plt.show()
    
    if predicted_class == 'Benign':
        interpretation = "Tế bào máu bình thường, không có dấu hiệu của bệnh bạch cầu."
    else:
        interpretation = f"Tế bào có dấu hiệu của bệnh bạch cầu giai đoạn {predicted_class}."
    
    print(f"Phân tích: {interpretation}")
    
    return predicted_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dự đoán ảnh tế bào')
    parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                          'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                      default='resnet', help='Chọn mô hình để dự đoán')
    parser.add_argument('--weights', required=True, help='Đường dẫn tới file weights')
    parser.add_argument('--image', required=True, help='Đường dẫn tới ảnh cần dự đoán')
    
    args = parser.parse_args()
    
    predict_image(args.model, args.weights, args.image)
