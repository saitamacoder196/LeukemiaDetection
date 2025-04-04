#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script đánh giá mô hình phát hiện bệnh qua ảnh tế bào.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_data
from utils.visualization import plot_confusion_matrix, plot_roc_curve

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

def evaluate_model(model_name, weights_path):
    """
    Đánh giá mô hình phát hiện bệnh qua ảnh tế bào.
    
    Args:
        model_name (str): Tên mô hình cần đánh giá.
        weights_path (str): Đường dẫn tới file weights.
        
    Returns:
        dict: Các chỉ số đánh giá của mô hình.
    """
    print(f"Đang đánh giá mô hình {model_name} với weights từ {weights_path}...")
    
    # Tiền xử lý dữ liệu
    _, X_test, _, y_test = preprocess_data(test_split=0.3)
    
    # Tải mô hình
    model = load_model(model_name, weights_path)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Tính các chỉ số đánh giá
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join('results', 'metrics', f'{model_name}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Classification report
    class_names = ['Benign', 'Early Pre B', 'Pre B', 'Pro B']
    report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # ROC curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve((y_test[:, i] > 0.5).astype(int), y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plot_roc_curve(fpr, tpr, roc_auc, class_names, os.path.join(results_dir, 'roc_curve.png'))
    
    # Lưu các chỉ số đánh giá
    metrics = {
        'accuracy': np.mean(y_pred_classes == y_test_classes),
        'roc_auc': roc_auc
    }
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("ROC AUC:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    
    print(f"\nĐã lưu kết quả đánh giá tại: {results_dir}")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Đánh giá mô hình phát hiện bệnh qua ảnh tế bào')
    parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                          'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                      default='resnet', help='Chọn mô hình để đánh giá')
    parser.add_argument('--weights', required=True, help='Đường dẫn tới file weights')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.weights)
