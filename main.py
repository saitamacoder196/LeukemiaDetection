#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Điểm vào chính của dự án phát hiện bệnh qua ảnh tế bào.
"""

import argparse
import os
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.predict import predict_image

def main():
    """Hàm chính của chương trình."""
    parser = argparse.ArgumentParser(description='Phát hiện bệnh qua ảnh tế bào')
    
    subparsers = parser.add_subparsers(dest='command', help='Lệnh cần thực hiện')
    
    # Parser cho lệnh train
    train_parser = subparsers.add_parser('train', help='Huấn luyện mô hình')
    train_parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                                'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                            default='resnet', help='Chọn mô hình để huấn luyện')
    train_parser.add_argument('--epochs', type=int, default=100, help='Số lượng epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    
    # Parser cho lệnh evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Đánh giá mô hình')
    eval_parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                               'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                           default='resnet', help='Chọn mô hình để đánh giá')
    eval_parser.add_argument('--weights', required=True, help='Đường dẫn tới file weights')
    
    # Parser cho lệnh predict
    predict_parser = subparsers.add_parser('predict', help='Dự đoán ảnh')
    predict_parser.add_argument('--model', choices=['resnet', 'efficientnet', 'inception', 'xception', 
                                                  'regnetx', 'convnext', 'alexnet', 'vgg', 'yolo'], 
                              default='resnet', help='Chọn mô hình để dự đoán')
    predict_parser.add_argument('--weights', required=True, help='Đường dẫn tới file weights')
    predict_parser.add_argument('--image', required=True, help='Đường dẫn tới ảnh cần dự đoán')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.model, args.epochs, args.batch_size)
    elif args.command == 'evaluate':
        evaluate_model(args.model, args.weights)
    elif args.command == 'predict':
        predict_image(args.model, args.weights, args.image)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
