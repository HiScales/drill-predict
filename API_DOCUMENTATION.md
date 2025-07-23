# 钻孔预测 API 文档

## 概述

这是一个基于Flask的RESTful API服务，提供智能钻孔预测功能。该服务使用智能网格优先算法，结合建筑物特征点吸附，为工程项目生成最优的钻孔方案。

## 基础信息

- **服务地址**: `http://localhost:5000`
- **API版本**: v1
- **数据格式**: JSON
- **文件上传**: 支持Excel (.xlsx) 格式

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行服务
```bash
python app.py
```

服务将在 `http://localhost:5000` 启动

## API 接口

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应示例**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "service": "Drill Prediction API"
}
```

### 2. 钻孔预测

**POST** `/api/v1/predict`

上传Excel文件进行钻孔预测。

**请求参数**:
- `file`: Excel文件 (multipart/form-data)

**响应示例**:
```json
{
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "prediction_count": 42,
    "spacing_constraints": {
        "min_spacing": 15.0,
        "max_spacing": 30.0
    },
    "site_info": {
        "area": 13207.1,
        "perimeter": 461.9,
        "building_count": 4
    },
    "grid_info": {
        "rotation_angle_degrees": 0.0,
        "grid_candidates_count": 1810
    },
    "download_urls": {
        "prediction_excel": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000/excel",
        "visualization_image": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000/image"
    },
    "prediction_points": [
        [550872.3, 372748.5],
        [550992.3, 372838.5]
    ]
}
```

### 3. 预测与对比

**POST** `/api/v1/predict/compare`

上传测试文件和验证文件，进行预测并与真实数据对比。

**请求参数**:
- `test_file`: 测试Excel文件 (multipart/form-data)
- `validation_file`: 验证Excel文件 (multipart/form-data)

**响应示例**:
```json
{
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "prediction_count": 42,
    "real_count": 29,
    "spacing_constraints": {
        "min_spacing": 15.0,
        "max_spacing": 30.0
    },
    "site_info": {
        "area": 13207.1,
        "perimeter": 461.9,
        "building_count": 4
    },
    "grid_info": {
        "rotation_angle_degrees": 0.0,
        "grid_candidates_count": 1810
    },
    "accuracy_metrics": {
        "average_distance_error": 11.5,
        "max_distance_error": 27.3,
        "min_distance_error": 0.0,
        "distance_error_std": 8.2
    },
    "download_urls": {
        "prediction_excel": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000/excel",
        "comparison_image": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000/compare"
    },
    "prediction_points": [[550872.3, 372748.5]],
    "real_points": [[550873.6, 372750.0]]
}
```

### 4. 文件下载

**GET** `/api/v1/download/{file_id}/{file_type}`

下载预测结果文件。

**路径参数**:
- `file_id`: 文件ID
- `file_type`: 文件类型 (`excel`, `image`, `compare`)

**文件类型说明**:
- `excel`: 预测结果Excel文件
- `image`: 可视化图片
- `compare`: 对比图片

### 5. 批量预测

**POST** `/api/v1/batch-predict`

批量处理多个Excel文件。

**请求参数**:
- `files`: 多个Excel文件 (multipart/form-data)

**响应示例**:
```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "total_files": 5,
    "successful_predictions": 4,
    "failed_predictions": 1,
    "results": [
        {
            "original_filename": "project_1.xlsx",
            "file_id": "550e8400-e29b-41d4-a716-446655440000_0",
            "prediction_count": 42,
            "site_area": 13207.1,
            "building_count": 4,
            "download_urls": {
                "prediction_excel": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000_0/excel",
                "visualization_image": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000_0/image"
            }
        }
    ]
}
```

### 6. 服务状态

**GET** `/api/v1/status`

获取服务运行状态和统计信息。

**响应示例**:
```json
{
    "status": "running",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "uploaded_files": 25,
    "generated_results": 25,
    "upload_folder": "uploads",
    "result_folder": "results"
}
```

## Excel文件格式要求

### 必需的工作表

1. **Site工作表**: 包含场地边界坐标
   - 列名: `X`, `Y`
   - 数据: 场地边界的坐标点

2. **Building工作表**: 包含建筑物信息 (可选)
   - 列名: `Group`, `X`, `Y`
   - 数据: 建筑物轮廓坐标点

3. **Drill工作表**: 包含钻孔信息 (验证文件需要)
   - 列名: `X`, `Y`
   - 数据: 钻孔点坐标

4. **Spacing工作表**: 包含间距约束
   - 单元格A1: 最小间距,最大间距 (如: "15,30")

### 示例Excel结构

```
Site工作表:
| X       | Y       |
|---------|---------|
| 100.0   | 200.0   |
| 150.0   | 200.0   |
| 150.0   | 250.0   |
| 100.0   | 250.0   |

Building工作表:
| Group | X       | Y       |
|-------|---------|---------|
| 1     | 120.0   | 220.0   |
| 1     | 130.0   | 220.0   |
| 1     | 130.0   | 230.0   |
| 1     | 120.0   | 230.0   |

Spacing工作表:
| A1        |
|-----------|
| 15,30     |
```

## 错误处理

### 常见错误码

- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

### 错误响应格式

```json
{
    "error": "错误类型",
    "message": "详细错误信息"
}
```

## 使用示例

### Python客户端示例

```python
import requests

# 健康检查
response = requests.get('http://localhost:5000/health')
print(response.json())

# 钻孔预测
with open('project.xlsx', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/v1/predict', files=files)
    result = response.json()
    print(f"预测完成，生成了 {result['prediction_count']} 个钻孔点")

# 下载预测结果
file_id = result['file_id']
response = requests.get(f'http://localhost:5000/api/v1/download/{file_id}/excel')
with open('prediction_result.xlsx', 'wb') as f:
    f.write(response.content)
```

### JavaScript客户端示例

```javascript
// 健康检查
fetch('http://localhost:5000/health')
    .then(response => response.json())
    .then(data => console.log(data));

// 钻孔预测
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/api/v1/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`预测完成，生成了 ${data.prediction_count} 个钻孔点`);
    // 下载预测结果
    window.open(`http://localhost:5000/api/v1/download/${data.file_id}/excel`);
});
```

## 算法说明

### 智能网格优先算法

1. **场地方向分析**: 计算场地主方向角度
2. **旋转网格生成**: 根据主方向生成旋转网格候选点
3. **点吸附处理**: 将靠近建筑物特征的点吸附到角点或边线
4. **间距约束筛选**: 确保最终点满足最小间距要求

### 预测精度评估

- **平均距离误差**: 预测点到最近真实点的平均距离
- **最大距离误差**: 预测点到最近真实点的最大距离
- **最小距离误差**: 预测点到最近真实点的最小距离
- **距离误差标准差**: 距离误差的标准差

## 注意事项

1. 上传文件大小限制为16MB
2. 支持的文件格式为Excel (.xlsx)
3. 服务会自动创建uploads和results文件夹
4. 生成的文件会保留在服务器上，建议定期清理
5. 生产环境部署时请配置适当的安全措施 