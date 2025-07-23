#!/usr/bin/env python3
"""
测试角点间距约束的脚本
"""

import numpy as np
from shapely.geometry import Polygon, Point
from main import select_uniform_grid_points_with_priority, get_polygons, read_project_xlsx

def test_corner_constraint():
    """测试角点间距约束"""
    print("测试角点间距约束")
    print("=" * 50)
    
    # 测试有建筑物的项目
    test_projects = [27, 29]  # 这些项目有建筑物
    
    for project_id in test_projects:
        try:
            print(f"\n测试项目 {project_id}")
            print("-" * 30)
            
            # 读取项目数据
            test_proj = read_project_xlsx(f'test/{project_id}.xlsx')
            site_poly, buildings = get_polygons(test_proj['site'], test_proj['building'])
            min_spacing, max_spacing = test_proj['spacing']
            
            print(f"建筑物数量: {len(buildings)}")
            
            # 收集所有角点
            all_corners = []
            for b in buildings:
                coords = list(b.exterior.coords)
                for corner in coords[:-1]:
                    if site_poly.contains(Point(corner)):
                        all_corners.append(tuple(corner))
            
            print(f"原始角点数量: {len(all_corners)}")
            
            # 手动应用角点间距约束
            valid_corners = []
            for i, corner in enumerate(all_corners):
                too_close = False
                for selected_corner in valid_corners:
                    dist = np.linalg.norm(np.array(corner) - np.array(selected_corner))
                    if dist < 5:
                        too_close = True
                        print(f"  角点 {i+1}: {corner} 距离 {selected_corner} 太近 ({dist:.2f}m < 5m)")
                        break
                if not too_close:
                    valid_corners.append(corner)
                    print(f"  选择角点 {i+1}: {corner}")
            
            print(f"约束后角点数量: {len(valid_corners)}")
            
            # 运行完整的预测算法
            print(f"\n运行完整预测算法...")
            pred_drills = select_uniform_grid_points_with_priority(site_poly, buildings, min_spacing, max_spacing)
            print(f"最终预测点数: {len(pred_drills)}")
            
        except Exception as e:
            print(f"项目 {project_id} 测试失败: {e}")

if __name__ == "__main__":
    test_corner_constraint() 