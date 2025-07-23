# 钻孔预测 API 服务

基于Flask的RESTful API服务，提供智能钻孔预测功能。使用智能网格优先算法，结合建筑物特征点吸附，为工程项目生成最优的钻孔方案。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式1: 直接启动
python app.py

# 方式2: 使用启动脚本
python start_server.py
```

服务将在 `http://localhost:5000` 启动

### 3. 测试API

```bash
python test_client.py
```

## 📁 项目结构

```
├── app.py                 # Flask主应用
├── main_1.py             # 钻孔预测核心算法
├── requirements.txt      # Python依赖
├── API_DOCUMENTATION.md  # API详细文档
├── test_client.py        # 测试客户端
├── start_server.py       # 服务器启动脚本
├── README.md            # 项目说明
├── uploads/             # 上传文件目录
├── results/             # 结果文件目录
├── test/                # 测试数据
├── validation/          # 验证数据
└── result1/             # 预测结果
```

## 🔧 API 接口

### 基础接口

- `GET /health` - 健康检查
- `GET /api/v1/status` - 服务状态

### 核心功能

- `POST /api/v1/predict` - 钻孔预测
- `POST /api/v1/predict/compare` - 预测与对比
- `POST /api/v1/batch-predict` - 批量预测
- `GET /api/v1/download/{file_id}/{file_type}` - 文件下载

## 📊 功能特点

### 🎯 智能算法
- **场地方向分析**: 自动计算场地主方向
- **旋转网格生成**: 根据主方向生成最优网格
- **点吸附处理**: 智能吸附到建筑物特征点
- **间距约束**: 确保满足工程间距要求

### 📈 预测精度
- **平均距离误差**: 预测点到真实点的平均距离
- **最大距离误差**: 预测点到真实点的最大距离
- **精度统计**: 详细的误差分析报告

### 🖼️ 可视化
- **网格线显示**: 可视化生成的网格系统
- **建筑物标注**: 显示场地内建筑物
- **钻孔点对比**: 预测点与真实点的对比图
- **高清输出**: 300DPI高质量图片

## 📋 Excel文件格式

### 必需工作表

1. **Site工作表**: 场地边界坐标
   ```
   | X       | Y       |
   |---------|---------|
   | 100.0   | 200.0   |
   | 150.0   | 200.0   |
   ```

2. **Building工作表**: 建筑物信息 (可选)
   ```
   | Group | X       | Y       |
   |-------|---------|---------|
   | 1     | 120.0   | 220.0   |
   | 1     | 130.0   | 220.0   |
   ```

3. **Spacing工作表**: 间距约束
   ```
   | A1        |
   |-----------|
   | 15,30     |
   ```

## 🧪 使用示例

### Python客户端

```python
import requests

# 钻孔预测
with open('project.xlsx', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/v1/predict', files=files)
    result = response.json()
    print(f"预测完成，生成了 {result['prediction_count']} 个钻孔点")
```

### JavaScript客户端

```javascript
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
});
```

## 🔍 算法原理

### 智能网格优先算法

1. **场地方向计算**
   - 分析场地边界的所有边
   - 计算加权平均方向角度
   - 确定最优网格方向

2. **旋转网格生成**
   - 根据主方向生成旋转网格
   - 过滤场地内的候选点
   - 动态添加中点补充

3. **点吸附处理**
   - 优先吸附到建筑物角点 (距离≤8m)
   - 其次吸附到建筑物边线 (距离≤5m)
   - 保持原始点位置

4. **间距约束筛选**
   - 确保最小间距要求
   - 去重处理
   - 最终点选择

## 📊 性能指标

- **处理速度**: 单个项目 < 5秒
- **预测精度**: 平均误差 < 15米
- **文件大小**: 支持最大16MB
- **并发处理**: 支持批量预测

## 🛠️ 开发环境

- Python 3.8+
- Flask 2.3.3
- pandas 2.0.3
- numpy 1.24.3
- shapely 2.0.1
- matplotlib 3.7.2

## 📝 更新日志

### v1.0.0 (2024-01-15)
- ✅ 实现智能网格优先算法
- ✅ 添加点吸附功能
- ✅ 完整的RESTful API
- ✅ 批量处理支持
- ✅ 可视化功能
- ✅ 精度评估

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请提交 Issue 或联系开发团队。

---

**注意**: 生产环境部署时请配置适当的安全措施，如HTTPS、身份验证等。 