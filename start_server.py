#!/usr/bin/env python3
"""
钻孔预测API服务器启动脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'flask', 'flask-cors', 'pandas', 'numpy', 
        'shapely', 'matplotlib', 'openpyxl', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    directories = ['uploads', 'results']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ 目录已创建: {directory}")

def start_server():
    """启动服务器"""
    print("钻孔预测API服务器启动中...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 创建目录
    create_directories()
    
    # 启动服务器
    try:
        print("\n启动Flask服务器...")
        print("服务地址: http://localhost:5000")
        print("API文档: http://localhost:5000/health")
        print("按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        # 启动Flask应用
        subprocess.run([sys.executable, "app.py"])
        
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_server() 