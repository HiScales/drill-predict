#!/usr/bin/env python3
"""
钻孔预测API测试客户端
用于测试Flask API的各项功能
"""

import requests
import os
import json
from datetime import datetime

class DrillPredictionClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"健康检查失败: {e}")
            return None
    
    def predict_drilling(self, file_path):
        """钻孔预测"""
        try:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/api/v1/predict", files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"预测失败: {e}")
            return None
    
    def predict_and_compare(self, test_file_path, validation_file_path):
        """预测与对比"""
        try:
            if not os.path.exists(test_file_path) or not os.path.exists(validation_file_path):
                print(f"文件不存在: {test_file_path} 或 {validation_file_path}")
                return None
            
            with open(test_file_path, 'rb') as test_f, open(validation_file_path, 'rb') as val_f:
                files = {
                    'test_file': test_f,
                    'validation_file': val_f
                }
                response = self.session.post(f"{self.base_url}/api/v1/predict/compare", files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"预测对比失败: {e}")
            return None
    
    def batch_predict(self, file_paths):
        """批量预测"""
        try:
            files = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue
                files.append(('files', open(file_path, 'rb')))
            
            if not files:
                print("没有有效的文件")
                return None
            
            response = self.session.post(f"{self.base_url}/api/v1/batch-predict", files=files)
            response.raise_for_status()
            
            # 关闭文件
            for _, f in files:
                f.close()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批量预测失败: {e}")
            return None
    
    def download_file(self, file_id, file_type, save_path):
        """下载文件"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/download/{file_id}/{file_type}")
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"文件已下载到: {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")
            return False
    
    def get_status(self):
        """获取服务状态"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取状态失败: {e}")
            return None

def print_json(data, title=""):
    """美化打印JSON数据"""
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

def main():
    """主测试函数"""
    client = DrillPredictionClient()
    
    print("钻孔预测API测试客户端")
    print("=" * 50)
    
    # 1. 健康检查
    print("\n1. 健康检查")
    health = client.health_check()
    if health:
        print_json(health, "健康检查结果")
    else:
        print("服务不可用，请检查服务是否启动")
        return
    
    # 2. 获取服务状态
    print("\n2. 服务状态")
    status = client.get_status()
    if status:
        print_json(status, "服务状态")
    
    # 3. 测试文件路径
    test_files = []
    for i in range(21, 30):
        test_file = f"test/{i}.xlsx"
        if os.path.exists(test_file):
            test_files.append(test_file)
    
    if not test_files:
        print("未找到测试文件，请确保test目录下有Excel文件")
        return
    
    # 4. 单个文件预测测试
    print(f"\n3. 单个文件预测测试 (使用 {test_files[0]})")
    result = client.predict_drilling(test_files[0])
    if result:
        print_json(result, "预测结果")
        
        # 下载预测结果
        file_id = result['file_id']
        client.download_file(file_id, 'excel', f'downloaded_prediction_{file_id}.xlsx')
        client.download_file(file_id, 'image', f'downloaded_visualization_{file_id}.png')
    
    # 5. 预测对比测试
    print(f"\n4. 预测对比测试")
    validation_file = f"validation/{test_files[0].split('/')[-1]}"
    if os.path.exists(validation_file):
        compare_result = client.predict_and_compare(test_files[0], validation_file)
        if compare_result:
            print_json(compare_result, "对比结果")
            
            # 下载对比图
            file_id = compare_result['file_id']
            client.download_file(file_id, 'compare', f'downloaded_comparison_{file_id}.png')
    else:
        print(f"验证文件不存在: {validation_file}")
    
    # 6. 批量预测测试
    print(f"\n5. 批量预测测试 (处理 {len(test_files[:3])} 个文件)")
    batch_result = client.batch_predict(test_files[:3])
    if batch_result:
        print_json(batch_result, "批量预测结果")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 