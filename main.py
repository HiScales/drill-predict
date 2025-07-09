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

def extract_features_for_point(pt, site_poly, buildings, spacing, max_dist_building=9999, keypoints=None, edge_points=None):
    point = Point(pt)
    dist_site = site_poly.exterior.distance(point)
    in_site = int(site_poly.contains(point))
    if buildings and len(buildings) > 0:
        dist_building = min(b.exterior.distance(point) for b in buildings)
        in_building = int(any(b.contains(point) for b in buildings))
    else:
        dist_building = max_dist_building
        in_building = 0
    min_spacing, max_spacing = spacing
    near_corner = 0
    near_edge = 0
    if keypoints is not None and len(keypoints) > 0:
        if np.min(np.linalg.norm(keypoints - np.array(pt), axis=1)) < 2:
            near_corner = 1
    if edge_points is not None and len(edge_points) > 0:
        if np.min(np.linalg.norm(edge_points - np.array(pt), axis=1)) < 2:
            near_edge = 1
    return [pt[0], pt[1], dist_site, in_site, dist_building, in_building, min_spacing, max_spacing, near_corner, near_edge]

def prepare_training_data(train_data, grid_step=2):
    X, y = [], []
    for idx, proj in enumerate(train_data):
        print(f"正在处理第{idx+1}个项目")
        try:
            site_poly, buildings = get_polygons(proj['site'], proj['building'])
            minx, miny, maxx, maxy = site_poly.bounds
            max_dist_building = max(maxx - minx, maxy - miny) * 2
            grid_x, grid_y = np.meshgrid(
                np.arange(minx, maxx, grid_step),
                np.arange(miny, maxy, grid_step)
            )
            drill_points = proj['drill'][['X', 'Y']].values
            for x, y_ in zip(grid_x.flatten(), grid_y.flatten()):
                pt = (x, y_)
                if not site_poly.contains(Point(pt)):
                    continue
                feat = extract_features_for_point(pt, site_poly, buildings, proj['spacing'], max_dist_building)
                label = int(np.any(np.linalg.norm(drill_points - np.array(pt), axis=1) < grid_step))
                X.append(feat)
                y.append(label)
        except Exception as e:
            print(f'第{idx+1}个项目处理异常，已跳过，异常信息：{e}')
    return np.array(X), np.array(y)

def train_drill_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    return clf

def generate_candidate_points_ml(site_poly, buildings, spacing, grid_step=5):
    minx, miny, maxx, maxy = site_poly.bounds
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, grid_step),
        np.arange(miny, maxy, grid_step)
    )
    candidates = []
    for x, y_ in zip(grid_x.flatten(), grid_y.flatten()):
        pt = (x, y_)
        if site_poly.contains(Point(pt)):
            candidates.append(pt)
    if buildings and len(buildings) > 0:
        keypoints, edge_points = get_building_keypoints_and_edges(buildings, edge_step=5)
        candidates = np.vstack([candidates, keypoints, edge_points])
        # 去重
        candidates = np.unique(candidates, axis=0)
    return np.array(candidates)

def select_points_with_spacing_constraint(candidates, probs, min_spacing, max_spacing, keypoints=None, edge_points=None):
    # 标记角点和边线点
    is_corner = np.zeros(len(candidates), dtype=bool)
    is_edge = np.zeros(len(candidates), dtype=bool)
    if keypoints is not None and len(keypoints) > 0:
        is_corner = np.min(np.linalg.norm(candidates[:, None, :] - keypoints[None, :, :], axis=2), axis=1) < 2
    if edge_points is not None and len(edge_points) > 0:
        is_edge = np.min(np.linalg.norm(candidates[:, None, :] - edge_points[None, :, :], axis=2), axis=1) < 2

    idx_sorted = np.argsort(-probs)
    selected = []
    # 1. 先选角点
    for idx in idx_sorted:
        if is_corner[idx]:
            pt = candidates[idx]
            if not selected or (all(np.linalg.norm(np.array(pt) - np.array(sel)) >= min_spacing for sel in selected) and min([np.linalg.norm(np.array(pt) - np.array(sel)) for sel in selected]) <= max_spacing):
                selected.append(pt)
    # 2. 再选边线点
    for idx in idx_sorted:
        if is_edge[idx] and not any(np.allclose(candidates[idx], s) for s in selected):
            pt = candidates[idx]
            if not selected or (all(np.linalg.norm(np.array(pt) - np.array(sel)) >= min_spacing for sel in selected) and min([np.linalg.norm(np.array(pt) - np.array(sel)) for sel in selected]) <= max_spacing):
                selected.append(pt)
    # 3. 最后选其他点
    for idx in idx_sorted:
        if not is_corner[idx] and not is_edge[idx] and not any(np.allclose(candidates[idx], s) for s in selected):
            pt = candidates[idx]
            if not selected or (all(np.linalg.norm(np.array(pt) - np.array(sel)) >= min_spacing for sel in selected) and min([np.linalg.norm(np.array(pt) - np.array(sel)) for sel in selected]) <= max_spacing):
                selected.append(pt)
    return np.array(selected)

# 可视化对比
def plot_drills(site_poly, buildings, real_drills, pred_drills, save_path):
    plt.figure(figsize=(8, 8))
    x, y = site_poly.exterior.xy
    plt.plot(x, y, 'k-', label='Site')
    
    # 只在有建筑物时绘制建筑物
    if buildings and len(buildings) > 0:
        for b in buildings:
            bx, by = b.exterior.xy
            plt.plot(bx, by, 'b-', label='Building')
    
    if real_drills is not None and len(real_drills) > 0:
        plt.scatter(real_drills[:, 0], real_drills[:, 1], c='g', marker='o', label='Real Drill')
    if pred_drills is not None and len(pred_drills) > 0:
        plt.scatter(pred_drills[:, 0], pred_drills[:, 1], c='r', marker='x', label='Pred Drill')
    
    plt.legend()
    plt.title('Drill Point Prediction vs Real')
    plt.savefig(save_path)
    plt.close()

# 导出预测结果
def save_pred_to_excel(pred_points, save_path):
    df = pd.DataFrame(pred_points, columns=['X', 'Y'])
    df.to_excel(save_path, index=False)

def get_building_keypoints_and_edges(buildings, edge_step=5):
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

def main():
    try:
        # 1. 读取训练数据
        print('开始读取训练数据...')
        train_data = read_all_projects('train')
        
        # 2. 生成训练集特征和标签
        print('正在生成训练特征...')
        X, y = prepare_training_data(train_data, grid_step=20)
        print(f'训练样本数: {len(X)}')
        
        # 3. 训练模型
        print('正在训练模型...')
        clf = train_drill_model(X, y)
        
        # 4. 预测test集
        print('开始处理测试数据...')
        test_data = read_all_projects('test')
        
        for i, proj in enumerate(test_data, start=26):
            print(f'正在处理第{i}个测试项目...')
            try:
                site_poly, buildings = get_polygons(proj['site'], proj['building'])
                minx, miny, maxx, maxy = site_poly.bounds
                max_dist_building = max(maxx - minx, maxy - miny) * 2
                candidates = generate_candidate_points_ml(site_poly, buildings, proj['spacing'], grid_step=5)
                X_test = np.array([
                    extract_features_for_point(pt, site_poly, buildings, proj['spacing'], max_dist_building)
                    for pt in candidates
                ])
                probs = clf.predict_proba(X_test)[:, 1]
                min_spacing, max_spacing = proj['spacing']
                pred_drills = select_points_with_spacing_constraint(candidates, probs, min_spacing, max_spacing)
                
                # 保存预测结果
                os.makedirs('result', exist_ok=True)
                save_pred_to_excel(pred_drills, f'result/{i}_pred.xlsx')
                
                # 可视化与真实对比
                try:
                    val_proj = read_project_xlsx(f'validation/{i}.xlsx')
                    real_drills = val_proj['drill'][['X', 'Y']].values
                    plot_drills(site_poly, buildings, real_drills, pred_drills, f'result/{i}_compare.png')
                except Exception as e:
                    print(f"Warning: 无法生成项目 {i} 的对比图: {str(e)}")
                    # 如果无法生成对比图，至少生成预测图
                    plot_drills(site_poly, buildings, None, pred_drills, f'result/{i}_pred.png')
                
                print(f'项目 {i} 处理完成')
                
            except Exception as e:
                print(f"处理项目 {i} 时发生错误: {str(e)}")
                continue
                
        print('全部预测与可视化已完成，结果保存在result目录下。')
    except Exception as e:
        print("发生异常：", e)

if __name__ == '__main__':
    main()