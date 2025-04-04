#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiện ích tiền xử lý dữ liệu.
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data():
    """
    Tải dữ liệu từ thư mục data/raw.
    
    Returns:
        tuple: (X, y) - Dữ liệu ảnh và nhãn.
    """
    data_dir = os.path.join('data', 'raw')
    classes = ['benign', 'early_pre_b', 'pre_b', 'pro_b']
    
    X = []
    y = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Thư mục {class_dir} không tồn tại. Đang bỏ qua...")
            continue
        
        for filename in os.listdir(class_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(i)
                except Exception as e:
                    print(f"Lỗi khi đọc ảnh {img_path}: {e}")
    
    return np.array(X), np.array(y)

def preprocess_data(test_split=0.3, img_size=(224, 224)):
    """
    Tiền xử lý dữ liệu: Chuẩn hóa, chia tập huấn luyện và kiểm tra.
    
    Args:
        test_split (float): Tỷ lệ dữ liệu dành cho kiểm tra.
        img_size (tuple): Kích thước ảnh đầu vào.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Dữ liệu đã được tiền xử lý.
    """
    # Tải dữ liệu
    X, y = load_data()
    
    if len(X) == 0:
        print("Không có dữ liệu để xử lý. Hãy kiểm tra thư mục data/raw.")
        return None, None, None, None
    
    # Resize và chuẩn hóa ảnh
    X_processed = []
    for img in X:
        img_resized = cv2.resize(img, img_size)
        img_normalized = img_resized / 255.0  # Chuẩn hóa về khoảng [0, 1]
        X_processed.append(img_normalized)
    
    X_processed = np.array(X_processed)
    
    # One-hot encoding nhãn
    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Chia tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_categorical, test_size=test_split, random_state=42, stratify=y_categorical
    )
    
    # Lưu dữ liệu đã xử lý
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    print(f"Dữ liệu đã được tiền xử lý và lưu tại: {processed_dir}")
    print(f"Kích thước tập huấn luyện: {X_train.shape}, {y_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def preprocess_image(image_path, img_size=(224, 224)):
    """
    Tiền xử lý một ảnh đơn lẻ.
    
    Args:
        image_path (str): Đường dẫn tới ảnh.
        img_size (tuple): Kích thước ảnh đầu vào.
        
    Returns:
        np.array: Ảnh đã được tiền xử lý.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size)
        img_normalized = img_resized / 255.0  # Chuẩn hóa về khoảng [0, 1]
        
        return img_normalized
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None
