import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 强制使用无界面后端

# 读取单个项目
def read_project_xlsx(file_path):
    try:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
        site = pd.read_excel(xls, sheet_name='Site')
        
        try:
            building = pd.read_excel(xls, sheet_name='Building')
            if building.empty or building.dropna().empty:
                building = pd.DataFrame(columns=['Group', 'X', 'Y'])
        except Exception as e:
            print(f"读取 {file_path} 的 Building sheet 失败: {e}")
            building = pd.DataFrame(columns=['Group', 'X', 'Y'])
            
        try:
            drill = pd.read_excel(xls, sheet_name='Drill')
        except Exception as e:
            print(f"读取 {file_path} 的 Drill sheet 失败: {e}")
            drill = pd.DataFrame(columns=['X', 'Y'])
            
        try:
            spacing = pd.read_excel(xls, sheet_name='Spacing', header=None).iloc[0, 0]
            spacing = [float(x) for x in str(spacing).split(',')]
        except Exception as e:
            print(f"读取 {file_path} 的 Spacing sheet 失败: {e}")
            spacing = [5, 50]  # 给个默认值
            
        return {
            'site': site,
            'building': building,
            'drill': drill,
            'spacing': spacing
        }
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        raise

# 读取所有项目
def read_all_projects(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.xlsx')], key=lambda x: int(x.split('.')[0]))
    projects = []
    for f in files:
        try:
            project = read_project_xlsx(os.path.join(folder, f))
            projects.append(project)
            print(f"成功读取项目: {f}")
        except Exception as e:
            print(f"读取项目 {f} 失败: {str(e)}")
    return projects

# 生成边界和建筑物多边形
def get_polygons(site_df, building_df):
    try:
        site_poly = Polygon(site_df[['X', 'Y']].values)
    except Exception as e:
        print(f"Error creating site polygon: {str(e)}")
        raise
        
    buildings = []
    if (
        building_df is not None and
        not building_df.empty and
        'Group' in building_df.columns and
        building_df.dropna(subset=['X', 'Y']).shape[0] > 0
    ):
        for group in building_df['Group'].dropna().unique():
            try:
                group_df = building_df[(building_df['Group'] == group) & building_df['X'].notna() & building_df['Y'].notna()]
                if len(group_df) >= 3:
                    buildings.append(Polygon(group_df[['X', 'Y']].values))
            except Exception as e:
                print(f"Warning: 跳过无效的建筑物组 {group}: {str(e)}")
                continue
    return site_poly, buildings

def get_building_keypoints_and_edges(buildings, edge_step=10):
    keypoints = []
    edge_points = []
    for b in buildings:
        coords = list(b.exterior.coords)
        keypoints.extend(coords[:-1])  # 最后一个点和第一个重复
        # 边线采样
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i+1])
            dist = np.linalg.norm(p2 - p1)
            n = max(1, int(dist // edge_step))
            for t in np.linspace(0, 1, n, endpoint=False):
                pt = p1 + t * (p2 - p1)
                edge_points.append(tuple(pt))
    return np.array(keypoints), np.array(edge_points)

def get_site_keypoints_and_edges(site_poly, edge_step=5):
    """提取site边界的角点和边线点"""
    coords = list(site_poly.exterior.coords)
    keypoints = coords[:-1]  # 去掉重复的最后一个点
    edge_points = []
    
    # site边界边线采样
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i+1])
        dist = np.linalg.norm(p2 - p1)
        n = max(1, int(dist // edge_step))
        for t in np.linspace(0, 1, n, endpoint=False):
            pt = p1 + t * (p2 - p1)
            edge_points.append(tuple(pt))
    
    return np.array(keypoints), np.array(edge_points)

def calculate_site_orientation(site_poly):
    """
    计算site的主方向角度，用于调整网格方向
    返回旋转角度（弧度）
    """
    try:
        # 获取site边界的所有边
        coords = list(site_poly.exterior.coords)[:-1]  # 去掉重复的最后一个点
        
        if len(coords) < 3:
            return 0  # 如果点数不够，返回0度
        
        # 计算所有边的方向角度和长度
        edge_angles = []
        edge_lengths = []
        
        for i in range(len(coords)):
            p1 = np.array(coords[i])
            p2 = np.array(coords[(i + 1) % len(coords)])
            
            # 计算边的向量和长度
            edge_vector = p2 - p1
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-6:  # 避免长度为0的边
                # 计算角度（归一化到0-π/2范围，因为我们只关心方向，不关心正负）
                angle = np.arctan2(edge_vector[1], edge_vector[0])
                # 将角度归一化到0-π/2范围（因为矩形网格只需要考虑两个垂直方向）
                angle = angle % (np.pi / 2)
                
                edge_angles.append(angle)
                edge_lengths.append(edge_length)
        
        if not edge_angles:
            return 0
        
        # 按边长加权，选择最主要的方向
        edge_angles = np.array(edge_angles)
        edge_lengths = np.array(edge_lengths)
        
        # 找出最长的几条边的平均方向
        # 按长度排序，取前几条最长的边
        sorted_indices = np.argsort(edge_lengths)[::-1]
        top_edges = min(4, len(sorted_indices))  # 最多考虑前4条边
        
        main_angles = edge_angles[sorted_indices[:top_edges]]
        main_lengths = edge_lengths[sorted_indices[:top_edges]]
        
        # 计算加权平均角度
        # 处理角度的周期性（接近0和π/2的角度应该被认为是相似的）
        cos_angles = np.cos(2 * main_angles)  # 乘以2是为了处理π/2的周期性
        sin_angles = np.sin(2 * main_angles)
        
        # 加权平均
        avg_cos = np.average(cos_angles, weights=main_lengths)
        avg_sin = np.average(sin_angles, weights=main_lengths)
        
        # 转换回角度
        avg_angle = np.arctan2(avg_sin, avg_cos) / 2
        
        # 确保角度在合理范围内
        avg_angle = avg_angle % (np.pi / 2)
        
        print(f"Site主方向角度: {np.degrees(avg_angle):.1f}度")
        return avg_angle
        
    except Exception as e:
        print(f"计算site方向时出错: {e}")
        return 0

def generate_rotated_grid(site_poly, grid_spacing, rotation_angle):
    """
    按指定角度生成旋转网格
    """
    # 获取site边界
    minx, miny, maxx, maxy = site_poly.bounds
    
    # 计算site中心点
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # 计算旋转后需要的网格范围（要足够大以覆盖旋转后的site）
    diagonal = np.sqrt((maxx - minx)**2 + (maxy - miny)**2)
    grid_range = diagonal * 1.2  # 留出余量
    
    # 生成基础正交网格
    steps = int(grid_range / grid_spacing) + 1
    grid_coords = []
    
    # 生成主网格点
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            # 基础网格点（相对于中心点）
            base_x = i * grid_spacing
            base_y = j * grid_spacing
            
            # 应用旋转变换
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)
            
            rotated_x = base_x * cos_theta - base_y * sin_theta
            rotated_y = base_x * sin_theta + base_y * cos_theta
            
            # 转换到实际坐标（相对于site中心）
            actual_x = center_x + rotated_x
            actual_y = center_y + rotated_y
            
            grid_coords.append((actual_x, actual_y))
    
    # 过滤出在site内的点
    grid_candidates = []
    for pt in grid_coords:
        if is_point_inside_site(pt, site_poly):
            grid_candidates.append(pt)
    
    # 动态添加中点
    additional_points = []
    for i in range(len(grid_candidates)):
        for j in range(i + 1, len(grid_candidates)):
            pt1 = np.array(grid_candidates[i])
            pt2 = np.array(grid_candidates[j])
            dist = np.linalg.norm(pt2 - pt1)
            
            # 如果两点之间距离过大，添加中点
            if dist > grid_spacing * 1.5:
                mid_point = (pt1 + pt2) / 2
                if is_point_inside_site(tuple(mid_point), site_poly):
                    additional_points.append(tuple(mid_point))
    
    grid_candidates.extend(additional_points)
    print(f"旋转网格生成完成: 角度{np.degrees(rotation_angle):.1f}度, 候选点{len(grid_candidates)}个")
    return grid_candidates

def is_point_inside_site(point, site_poly, buffer_distance=0):
    """
    严格检查点是否在site边界内
    buffer_distance: 负值表示向内缩进，正值表示向外扩展
    """
    if buffer_distance == 0:
        return site_poly.contains(Point(point))
    else:
        buffered_site = site_poly.buffer(buffer_distance)
        return buffered_site.contains(Point(point))

def snap_point_to_building_features(point, buildings, snap_distance_corner=8, snap_distance_edge=5):
    """
    将点吸附到建筑物特征（角点或边线），但必须确保吸附后仍在site边界内
    """
    if not buildings or len(buildings) == 0:
        return point
    
    original_point = np.array(point)
    best_point = original_point
    best_priority = 0  # 0: 无吸附, 1: 边线吸附, 2: 角点吸附
    
    try:
        # 获取建筑物角点和边线
        building_keypoints, building_edge_points = get_building_keypoints_and_edges(buildings, edge_step=1)
        
        # 尝试吸附到角点（最高优先级）
        if len(building_keypoints) > 0:
            distances = np.linalg.norm(building_keypoints - original_point, axis=1)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist <= snap_distance_corner:
                candidate_point = building_keypoints[min_dist_idx]
                best_point = candidate_point
                best_priority = 2
        
        # 如果没有角点吸附，尝试吸附到边线
        if best_priority < 2 and len(building_edge_points) > 0:
            distances = np.linalg.norm(building_edge_points - original_point, axis=1)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist <= snap_distance_edge:
                candidate_point = building_edge_points[min_dist_idx]
                best_point = candidate_point
                best_priority = 1
        
    except Exception as e:
        print(f"点吸附过程中出错: {e}")
        return point
    
    return tuple(best_point)

def select_grid_with_snap(site_poly, buildings, min_spacing, max_spacing):
    """
    智能网格优先 + 点吸附到建筑物特征
    """
    print(f"\n=== 项目详细信息 ===")
    print(f"Site面积: {site_poly.area:.1f}平方米")
    print(f"Site边界长度: {site_poly.length:.1f}米")
    print(f"Building数量: {len(buildings)}个")
    print(f"Spacing约束: [{min_spacing}, {max_spacing}]")
    
    # 第一步：生成智能网格点
    print(f"\n=== 生成智能网格 ===")
    rotation_angle = calculate_site_orientation(site_poly)
    # 网格间距默认为最大约束距离的0.8倍
    grid_spacing = max_spacing * 0.8
    print(f"网格间距: {grid_spacing:.1f}m (最大约束距离 {max_spacing}m 的 0.8 倍)")
    grid_candidates = generate_rotated_grid(site_poly, grid_spacing, rotation_angle)
    print(f"智能网格生成{len(grid_candidates)}个候选点")
    
    # 第二步：将网格点吸附到建筑物特征
    print(f"\n=== 点吸附到建筑物特征 ===")
    snapped_points = []
    corner_snaps = 0
    edge_snaps = 0
    
    for i, pt in enumerate(grid_candidates):
        original_pt = np.array(pt)
        snapped_pt = snap_point_to_building_features(pt, buildings)
        snapped_points.append(snapped_pt)
        
        # 统计吸附情况
        if not np.allclose(original_pt, snapped_pt, atol=1e-2):
            # 检查是角点吸附还是边线吸附
            building_keypoints, building_edge_points = get_building_keypoints_and_edges(buildings, edge_step=1)
            if len(building_keypoints) > 0:
                corner_dist = min(np.linalg.norm(building_keypoints - snapped_pt, axis=1))
                if corner_dist < 1e-2:
                    corner_snaps += 1
                else:
                    edge_snaps += 1
    
    print(f"角点吸附: {corner_snaps}个点")
    print(f"边线吸附: {edge_snaps}个点")
    print(f"无吸附: {len(snapped_points) - corner_snaps - edge_snaps}个点")
    
    # 第三步：去重并按间距约束筛选
    print(f"\n=== 去重和间距约束筛选 ===")
    unique_points = []
    for pt in snapped_points:
        # 检查是否与已选点距离过近
        if any(np.linalg.norm(np.array(pt) - np.array(up)) < min_spacing * 0.95 for up in unique_points):
            continue
        unique_points.append(pt)
    
    print(f"最终选择 {len(unique_points)} 个点")
    
    # 统计最终结果
    if len(unique_points) > 1:
        print(f"\n=== 间距分布统计 ===")
        all_distances = []
        for i in range(len(unique_points)):
            for j in range(i + 1, len(unique_points)):
                dist = np.linalg.norm(np.array(unique_points[i]) - np.array(unique_points[j]))
                all_distances.append(dist)
        if all_distances:
            print(f"最小间距: {min(all_distances):.1f}m")
            print(f"最大间距: {max(all_distances):.1f}m")
            print(f"平均间距: {np.mean(all_distances):.1f}m")
            print(f"间距标准差: {np.std(all_distances):.1f}m")
            violations = sum(1 for d in all_distances if d < min_spacing)
            print(f"最小间距违规: {violations}个 ({violations/len(all_distances)*100:.1f}%)")
    
    return np.array(unique_points), grid_candidates, rotation_angle, grid_spacing

def plot_drills(site_poly, buildings, real_drills, pred_drills, save_path, grid_candidates=None, rotation_angle=0, grid_spacing=None):
    """带网格线的对比可视化 - 优化比例"""
    plt.figure(figsize=(14, 12))
    
    # 绘制site边界
    x, y = site_poly.exterior.xy
    plt.plot(x, y, 'k-', linewidth=3, label='Site Boundary')
    
    # 绘制建筑物
    if buildings and len(buildings) > 0:
        for i, b in enumerate(buildings):
            bx, by = b.exterior.xy
            plt.plot(bx, by, 'blue', linewidth=2, label='Building' if i == 0 else "")
    
    # 绘制布孔网格线（如果提供了网格参数）
    if grid_spacing is not None:
        # 获取site边界
        minx, miny, maxx, maxy = site_poly.bounds
        
        # 计算site中心点
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        
        # 计算旋转后需要的网格范围（要足够大以覆盖旋转后的site）
        diagonal = np.sqrt((maxx - minx)**2 + (maxy - miny)**2)
        grid_range = diagonal * 1.2  # 留出余量
        
        # 生成网格线
        steps = int(grid_range / grid_spacing) + 1
        
        # 绘制网格线
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        # 绘制水平网格线
        for i in range(-steps, steps + 1):
            # 生成水平线的两个端点
            base_x1 = -grid_range
            base_y1 = i * grid_spacing
            base_x2 = grid_range
            base_y2 = i * grid_spacing
            
            # 应用旋转变换
            x1 = center_x + (base_x1 * cos_theta - base_y1 * sin_theta)
            y1 = center_y + (base_x1 * sin_theta + base_y1 * cos_theta)
            x2 = center_x + (base_x2 * cos_theta - base_y2 * sin_theta)
            y2 = center_y + (base_x2 * sin_theta + base_y2 * cos_theta)
            
            # 增强网格线可见性
            plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=1.2)
        
        # 绘制垂直网格线
        for j in range(-steps, steps + 1):
            # 生成垂直线的两个端点
            base_x1 = j * grid_spacing
            base_y1 = -grid_range
            base_x2 = j * grid_spacing
            base_y2 = grid_range
            
            # 应用旋转变换
            x1 = center_x + (base_x1 * cos_theta - base_y1 * sin_theta)
            y1 = center_y + (base_x1 * sin_theta + base_y1 * cos_theta)
            x2 = center_x + (base_x2 * cos_theta - base_y2 * sin_theta)
            y2 = center_y + (base_x2 * sin_theta + base_y2 * cos_theta)
            
            # 增强网格线可见性
            plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=1.2)
        
        # 绘制网格候选点（如果提供）- 调整大小和透明度
        if grid_candidates is not None and len(grid_candidates) > 0:
            grid_x = [pt[0] for pt in grid_candidates]
            grid_y = [pt[1] for pt in grid_candidates]
            plt.scatter(grid_x, grid_y, c='lightblue', marker='.', s=20, alpha=0.5, label='Grid Candidates')
        
        # 添加网格线说明
        plt.text(0.02, 0.98, f'Grid Spacing: {grid_spacing}m', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 绘制真实钻孔点 - 调整大小
    if real_drills is not None and len(real_drills) > 0:
        plt.scatter(real_drills[:, 0], real_drills[:, 1], c='green', marker='o', s=80, 
                   edgecolors='darkgreen', linewidth=1.5, label='Real Drill Points', zorder=5)
    
    # 绘制预测钻孔点 - 调整大小
    if pred_drills is not None and len(pred_drills) > 0:
        plt.scatter(pred_drills[:, 0], pred_drills[:, 1], c='red', marker='x', s=120, 
                   linewidth=2.5, label='Predicted Drill Points', zorder=6)
    
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    plt.title('Drill Point Prediction with Grid Visualization (Smart Grid + Snap)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.axis('equal')
    
    # 设置坐标轴范围，留出适当边距
    margin = grid_spacing * 2 if grid_spacing else 10
    plt.xlim(minx - margin, maxx + margin)
    plt.ylim(miny - margin, maxy + margin)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_pred_to_excel(pred_points, save_path):
    df = pd.DataFrame(pred_points, columns=['X', 'Y'])
    df.to_excel(save_path, index=False)

def main():
    try:
        # 处理所有测试数据
        print('开始处理所有测试数据...')
        test_data = read_all_projects('test')
        
        # 获取所有项目ID
        test_files = sorted([f for f in os.listdir('test') if f.endswith('.xlsx')], 
                           key=lambda x: int(x.split('.')[0]))
        project_ids = [int(f.split('.')[0]) for f in test_files]
        
        print(f'找到 {len(project_ids)} 个项目: {project_ids}')
        
        for project_id in project_ids:
            print(f'\n正在处理项目 {project_id}...')
            try:
                # 读取test数据
                test_proj = read_project_xlsx(f'test/{project_id}.xlsx')
                site_poly, buildings = get_polygons(test_proj['site'], test_proj['building'])
                min_spacing, max_spacing = test_proj['spacing']
                
                # 使用智能网格优先 + 点吸附算法
                print(f'项目 {project_id} 开始智能网格优先 + 点吸附布孔...')
                
                # 执行智能选择
                pred_drills, grid_candidates, rotation_angle, grid_spacing = select_grid_with_snap(site_poly, buildings, min_spacing, max_spacing)
                print(f'项目 {project_id} 布孔完成，共选择{len(pred_drills)}个点')
                
                # 保存预测结果
                os.makedirs('result1', exist_ok=True)
                save_pred_to_excel(pred_drills, f'result1/{project_id}_pred.xlsx')
                
                # 可视化与真实对比
                try:
                    val_proj = read_project_xlsx(f'validation/{project_id}.xlsx')
                    real_drills = val_proj['drill'][['X', 'Y']].values
                    
                    # 打印对比信息
                    print(f'\n=== 项目 {project_id} 对比分析 ===')
                    print(f'预测孔位数量: {len(pred_drills)}')
                    print(f'真实孔位数量: {len(real_drills)}')
                    
                    if len(pred_drills) > 0:
                        print(f'预测孔位坐标范围:')
                        print(f'  X: {pred_drills[:, 0].min():.1f} - {pred_drills[:, 0].max():.1f}')
                        print(f'  Y: {pred_drills[:, 1].min():.1f} - {pred_drills[:, 1].max():.1f}')
                    
                    if len(real_drills) > 0:
                        print(f'真实孔位坐标范围:')
                        print(f'  X: {real_drills[:, 0].min():.1f} - {real_drills[:, 0].max():.1f}')
                        print(f'  Y: {real_drills[:, 1].min():.1f} - {real_drills[:, 1].max():.1f}')
                    
                    # 计算预测精度
                    if len(pred_drills) > 0 and len(real_drills) > 0:
                        # 计算每个预测点到最近真实点的距离
                        min_distances = []
                        for pred_pt in pred_drills:
                            distances = [np.linalg.norm(pred_pt - real_pt) for real_pt in real_drills]
                            min_distances.append(min(distances))
                        
                        if min_distances:
                            print(f'预测精度统计:')
                            print(f'  平均距离误差: {np.mean(min_distances):.1f}m')
                            print(f'  最大距离误差: {np.max(min_distances):.1f}m')
                            print(f'  最小距离误差: {np.min(min_distances):.1f}m')
                    
                    plot_drills(site_poly, buildings, real_drills, pred_drills, f'result1/{project_id}_compare.png', 
                               grid_candidates, rotation_angle, grid_spacing)
                    print(f'对比图已保存: result1/{project_id}_compare.png')
                    
                except Exception as e:
                    print(f"Warning: 无法生成项目 {project_id} 的对比图: {str(e)}")
                    plot_drills(site_poly, buildings, None, pred_drills, f'result1/{project_id}_pred.png', 
                               grid_candidates, rotation_angle, grid_spacing)
                    print(f'预测图已保存: result1/{project_id}_pred.png')
                
                print(f'项目 {project_id} 处理完成')
                
            except Exception as e:
                print(f"处理项目 {project_id} 时发生错误: {str(e)}")
                continue
                
        print('\n全部预测与可视化已完成，结果保存在result1目录下。')
        
        # 生成汇总报告
        print('\n=== 汇总报告 ===')
        result_files = [f for f in os.listdir('result1') if f.endswith('_pred.xlsx')]
        print(f'总共处理了 {len(result_files)} 个项目')
        
        total_pred_points = 0
        total_real_points = 0
        
        for file in sorted(result_files):
            project_id = file.split('_')[0]
            try:
                pred_df = pd.read_excel(f'result1/{file}')
                val_proj = read_project_xlsx(f'validation/{project_id}.xlsx')
                
                pred_count = len(pred_df)
                real_count = len(val_proj['drill'])
                
                total_pred_points += pred_count
                total_real_points += real_count
                
                print(f'项目 {project_id}: 预测 {pred_count} 个孔位, 真实 {real_count} 个孔位')
                
            except Exception as e:
                print(f'项目 {project_id}: 读取失败 - {str(e)}')
        
        print(f'\n总计:')
        print(f'预测孔位总数: {total_pred_points}')
        print(f'真实孔位总数: {total_real_points}')
        print(f'平均每个项目预测孔位: {total_pred_points/len(result_files):.1f}')
        print(f'平均每个项目真实孔位: {total_real_points/len(result_files):.1f}')
        
    except Exception as e:
        print("发生异常：", e)

if __name__ == '__main__':
    main()