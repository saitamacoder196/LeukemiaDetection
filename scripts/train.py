#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script huấn luyện mô hình phát hiện bệnh qua ảnh tế bào.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_data
from utils.augmentation import augment_data
from utils.evaluation import evaluate_model as eval_func
from utils.visualization import plot_training_history

def load_model(model_name):
    """
    Tải mô hình theo tên được chỉ định.
    
    Args:
        model_name (str): Tên mô hình cần tải.
        
    Returns:
        model: Mô hình đã tải.
    """
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
    
    return create_model()

def train_model(model_name, epochs=100, batch_size=32):
    """
    Huấn luyện mô hình phát hiện bệnh qua ảnh tế bào.
    
    Args:
        model_name (str): Tên mô hình cần huấn luyện.
        epochs (int): Số lượng epochs.
        batch_size (int): Kích thước batch.
        
    Returns:
        str: Đường dẫn tới file weights đã lưu.
    """
    print(f"Đang huấn luyện mô hình {model_name} với {epochs} epochs và batch size {batch_size}...")
    
    # Tiền xử lý dữ liệu
    X_train, X_val, y_train, y_val = preprocess_data()
    
    # Tăng cường dữ liệu
    X_train, y_train = augment_data(X_train, y_train)
    
    # Tải mô hình
    model = load_model(model_name)
    
    # Biên dịch mô hình
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Tạo thư mục để lưu weights
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join('results', 'model_checkpoints', f'{model_name}_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callback để lưu mô hình
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
    
    checkpoint_path = os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_accuracy:.4f}.h5')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    tensorboard_dir = os.path.join('results', 'logs', f'{model_name}_{timestamp}')
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback],
        verbose=1
    )
    
    # Lưu mô hình cuối cùng
    final_model_path = os.path.join(checkpoint_dir, f'{model_name}_final.h5')
    model.save(final_model_path)
    
    # Vẽ biểu đồ kết quả huấn luyện
    plot_training_history(history, model_name, timestamp)
    
    # Đánh giá mô hình
    eval_func(model, X_val, y_val, model_name, timestamp)
    
    print(f"Đã hoàn thành huấn luyện mô hình {model_name}.")
    print(f"Weights đã được lưu tại: {final_model_path}")
    
    return final_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phát hiện bệnh qua ảnh tế bào')
    parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                          'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                      default='resnet', help='Chọn mô hình để huấn luyện')
    parser.add_argument('--epochs', type=int, default=100, help='Số lượng epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    
    args = parser.parse_args()
    
    train_model(args.model, args.epochs, args.batch_size)
