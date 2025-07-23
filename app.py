from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 强制使用无界面后端
import tempfile
import uuid
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 导入钻孔预测的核心函数
from main_1 import (
    read_project_xlsx, get_polygons, calculate_site_orientation,
    generate_rotated_grid, snap_point_to_building_features,
    select_grid_with_snap, plot_drills, save_pred_to_excel,
    get_building_keypoints_and_edges, is_point_inside_site
)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Drill Prediction API'
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict_drilling():
    """
    钻孔预测接口
    接收Excel文件，返回预测结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'message': 'Please upload an Excel file'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # 检查文件格式
        if not file.filename.endswith('.xlsx'):
            return jsonify({
                'error': 'Invalid file format',
                'message': 'Please upload an Excel (.xlsx) file'
            }), 400
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.xlsx')
        
        # 保存上传的文件
        file.save(file_path)
        
        # 读取项目数据
        try:
            project_data = read_project_xlsx(file_path)
        except Exception as e:
            return jsonify({
                'error': 'File reading error',
                'message': f'Failed to read Excel file: {str(e)}'
            }), 400
        
        # 获取多边形和间距信息
        site_poly, buildings = get_polygons(project_data['site'], project_data['building'])
        min_spacing, max_spacing = project_data['spacing']
        
        # 执行钻孔预测
        pred_drills, grid_candidates, rotation_angle = select_grid_with_snap(
            site_poly, buildings, min_spacing, max_spacing
        )
        
        # 保存预测结果
        result_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_pred.xlsx')
        save_pred_to_excel(pred_drills, result_file)
        
        # 生成可视化图
        plot_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_visualization.png')
        plot_drills(site_poly, buildings, None, pred_drills, plot_file, 
                   grid_candidates, rotation_angle, min_spacing)
        
        # 准备响应数据
        response_data = {
            'file_id': file_id,
            'prediction_count': len(pred_drills),
            'spacing_constraints': {
                'min_spacing': min_spacing,
                'max_spacing': max_spacing
            },
            'site_info': {
                'area': float(site_poly.area),
                'perimeter': float(site_poly.length),
                'building_count': len(buildings)
            },
            'grid_info': {
                'rotation_angle_degrees': float(np.degrees(rotation_angle)),
                'grid_candidates_count': len(grid_candidates)
            },
            'download_urls': {
                'prediction_excel': f'/api/v1/download/{file_id}/excel',
                'visualization_image': f'/api/v1/download/{file_id}/image'
            },
            'prediction_points': pred_drills.tolist() if len(pred_drills) > 0 else []
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/v1/predict/compare', methods=['POST'])
def predict_and_compare():
    """
    钻孔预测与对比接口
    接收测试文件和验证文件，返回预测结果与真实数据的对比
    """
    try:
        # 检查文件上传
        if 'test_file' not in request.files or 'validation_file' not in request.files:
            return jsonify({
                'error': 'Missing files',
                'message': 'Please upload both test and validation Excel files'
            }), 400
        
        test_file = request.files['test_file']
        validation_file = request.files['validation_file']
        
        if test_file.filename == '' or validation_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select both test and validation files'
            }), 400
        
        # 检查文件格式
        if not test_file.filename.endswith('.xlsx') or not validation_file.filename.endswith('.xlsx'):
            return jsonify({
                'error': 'Invalid file format',
                'message': 'Please upload Excel (.xlsx) files'
            }), 400
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_test.xlsx')
        validation_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_validation.xlsx')
        
        # 保存文件
        test_file.save(test_path)
        validation_file.save(validation_path)
        
        # 读取数据
        try:
            test_data = read_project_xlsx(test_path)
            validation_data = read_project_xlsx(validation_path)
        except Exception as e:
            return jsonify({
                'error': 'File reading error',
                'message': f'Failed to read Excel files: {str(e)}'
            }), 400
        
        # 获取多边形和间距信息
        site_poly, buildings = get_polygons(test_data['site'], test_data['building'])
        min_spacing, max_spacing = test_data['spacing']
        
        # 执行钻孔预测
        pred_drills, grid_candidates, rotation_angle = select_grid_with_snap(
            site_poly, buildings, min_spacing, max_spacing
        )
        
        # 获取真实钻孔数据
        real_drills = validation_data['drill'][['X', 'Y']].values
        
        # 保存预测结果
        result_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_pred.xlsx')
        save_pred_to_excel(pred_drills, result_file)
        
        # 生成对比图
        compare_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_compare.png')
        plot_drills(site_poly, buildings, real_drills, pred_drills, compare_file, 
                   grid_candidates, rotation_angle, min_spacing)
        
        # 计算预测精度
        accuracy_metrics = {}
        if len(pred_drills) > 0 and len(real_drills) > 0:
            min_distances = []
            for pred_pt in pred_drills:
                distances = [np.linalg.norm(pred_pt - real_pt) for real_pt in real_drills]
                min_distances.append(min(distances))
            
            if min_distances:
                accuracy_metrics = {
                    'average_distance_error': float(np.mean(min_distances)),
                    'max_distance_error': float(np.max(min_distances)),
                    'min_distance_error': float(np.min(min_distances)),
                    'distance_error_std': float(np.std(min_distances))
                }
        
        # 准备响应数据
        response_data = {
            'file_id': file_id,
            'prediction_count': len(pred_drills),
            'real_count': len(real_drills),
            'spacing_constraints': {
                'min_spacing': min_spacing,
                'max_spacing': max_spacing
            },
            'site_info': {
                'area': float(site_poly.area),
                'perimeter': float(site_poly.length),
                'building_count': len(buildings)
            },
            'grid_info': {
                'rotation_angle_degrees': float(np.degrees(rotation_angle)),
                'grid_candidates_count': len(grid_candidates)
            },
            'accuracy_metrics': accuracy_metrics,
            'download_urls': {
                'prediction_excel': f'/api/v1/download/{file_id}/excel',
                'comparison_image': f'/api/v1/download/{file_id}/compare'
            },
            'prediction_points': pred_drills.tolist() if len(pred_drills) > 0 else [],
            'real_points': real_drills.tolist() if len(real_drills) > 0 else []
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction and comparison failed',
            'message': str(e)
        }), 500

@app.route('/api/v1/download/<file_id>/<file_type>', methods=['GET'])
def download_file(file_id, file_type):
    """
    文件下载接口
    """
    try:
        if file_type == 'excel':
            file_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_pred.xlsx')
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            return send_file(file_path, as_attachment=True, download_name=f'prediction_{file_id}.xlsx')
        
        elif file_type == 'image':
            file_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_visualization.png')
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            return send_file(file_path, as_attachment=True, download_name=f'visualization_{file_id}.png')
        
        elif file_type == 'compare':
            file_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_compare.png')
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            return send_file(file_path, as_attachment=True, download_name=f'comparison_{file_id}.png')
        
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({
            'error': 'Download failed',
            'message': str(e)
        }), 500

@app.route('/api/v1/batch-predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    接收包含多个项目的文件夹或压缩包
    """
    try:
        # 检查是否有文件上传
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files uploaded',
                'message': 'Please upload Excel files'
            }), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({
                'error': 'No files selected',
                'message': 'Please select files to upload'
            }), 400
        
        # 过滤Excel文件
        excel_files = [f for f in files if f.filename.endswith('.xlsx')]
        if not excel_files:
            return jsonify({
                'error': 'No Excel files found',
                'message': 'Please upload Excel (.xlsx) files'
            }), 400
        
        batch_id = str(uuid.uuid4())
        batch_results = []
        
        for file in excel_files:
            try:
                # 生成文件名
                file_id = f"{batch_id}_{len(batch_results)}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.xlsx')
                
                # 保存文件
                file.save(file_path)
                
                # 读取项目数据
                project_data = read_project_xlsx(file_path)
                site_poly, buildings = get_polygons(project_data['site'], project_data['building'])
                min_spacing, max_spacing = project_data['spacing']
                
                # 执行预测
                pred_drills, grid_candidates, rotation_angle = select_grid_with_snap(
                    site_poly, buildings, min_spacing, max_spacing
                )
                
                # 保存结果
                result_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_pred.xlsx')
                save_pred_to_excel(pred_drills, result_file)
                
                # 生成可视化
                plot_file = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_visualization.png')
                plot_drills(site_poly, buildings, None, pred_drills, plot_file, 
                           grid_candidates, rotation_angle, min_spacing)
                
                # 记录结果
                batch_results.append({
                    'original_filename': file.filename,
                    'file_id': file_id,
                    'prediction_count': len(pred_drills),
                    'site_area': float(site_poly.area),
                    'building_count': len(buildings),
                    'download_urls': {
                        'prediction_excel': f'/api/v1/download/{file_id}/excel',
                        'visualization_image': f'/api/v1/download/{file_id}/image'
                    }
                })
                
            except Exception as e:
                batch_results.append({
                    'original_filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_id': batch_id,
            'total_files': len(excel_files),
            'successful_predictions': len([r for r in batch_results if 'error' not in r]),
            'failed_predictions': len([r for r in batch_results if 'error' in r]),
            'results': batch_results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/v1/status', methods=['GET'])
def get_status():
    """
    服务状态接口
    """
    try:
        # 统计文件数量
        upload_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.xlsx')])
        result_count = len([f for f in os.listdir(app.config['RESULT_FOLDER']) if f.endswith('.xlsx')])
        
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'uploaded_files': upload_count,
            'generated_results': result_count,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'result_folder': app.config['RESULT_FOLDER']
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Status check failed',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 