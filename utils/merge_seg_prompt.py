"""
输入语义分割结果转换得到的seg.json，提示结果，输出合并后的结果
"""
import json
import argparse
from shapely.strtree import STRtree
from tqdm import tqdm
import sys,os
sys.path.append(os.path.abspath('./'))#add project root to PATH
from dataset.data_utils import cocoSeg2polygon
from shapely.geometry import Polygon
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_seg',default='dataset/loveda/Val/Rural/res/seg.json')
    parser.add_argument('--res_prompt_s',default='work_dir_loveda/small/res_prompt/results_polyon.json')
    parser.add_argument('--res_prompt_l',default='work_dir_loveda/large/res_prompt/results_polyon.json')
    parser.add_argument('--out',default='work_dir_loveda/merge_res.json')
    args = parser.parse_args()
    return args

def load_json(json_file,key=None):
    with open(json_file,'r') as f:
        data=json.load(f)
    if key:
        data=data[key]
    res={}
    for d in data:
        res[d['id']]=d
    return res
def merge_result(seg_res,prompt_small,prompt_large):
    seg_res=load_json(seg_res)#,key='annotations'
    prompt_small=load_json(prompt_small)
    prompt_large=load_json(prompt_large)
    merge_res=[]
    for k,v in seg_res.items():
        if k in prompt_small:
            res=prompt_small[k]
            res['add']=1
            merge_res.append(res)
        elif k in prompt_large:
            res=prompt_large[k]
            res['add']=1
            merge_res.append(res)
        else:
            merge_res.append(v)
    return merge_res

# 按 image_id 分组并将 segmentation 转换为 Polygon 对象
def group_by_image_id(merge_res):
    #merge_res: coco格式标注
    grouped_res = {}
    for item in merge_res:
        image_id = item['image_id']
        # 直接在这里创建 Polygon 对象
        item['polygon'] = cocoSeg2polygon(item['segmentation'])
        if image_id not in grouped_res:
            grouped_res[image_id] = []
        grouped_res[image_id].append(item)
    return grouped_res
def polygon2cocoSeg(polygon: Polygon):
    """
    将shapely Polygon对象转换为COCO格式的segmentation
    """
    if not polygon.is_valid:
        raise ValueError("Invalid Polygon")
    # 获取外边界的坐标，确保按顺时针顺序排列
    exterior_coords = list(polygon.exterior.coords)
    # 获取孔洞的坐标，按逆时针顺序排列
    interior_coords = [list(interior.coords) for interior in polygon.interiors]
    # 将坐标平铺到一维数组
    segmentation = [list(sum(exterior_coords, ()))]
    for interior in interior_coords:
        segmentation.append(list(sum(interior, ())))
    return segmentation
# 处理多边形重叠
def handle_polygon_overlap(grouped_res, n):
    """
    处理图像内带有 'add' 标记的多边形与原有多边形的重叠问题，并优化重叠判断过程。
    如果存在重叠区域，更新原有多边形为差集（除去和 'add' 多边形的交集）。
    
    参数：
    grouped_res (dict): 一个字典，键为 'image_id'，值为多边形列表，包含 'polygon' 键为 shapely Polygon 对象。
    n (int): seg_res的长度，用于唯一标识新增的多边形。
    
    返回:
    new_merge_res (list): 包含处理后多边形的列表。
    """
    # 合并处理后的结果
    new_merge_res = []
    # buffer_distance=3
    # area_low_thr={1:12.5, 2:15, 3:50, 4:15, 5:12.5, 6:12.5, 7:50, 8:30, 9:50}#各类面积阈值，小于阈值的多边形不要
    area_low_thr=12.5
    for image_id, polygons in tqdm(grouped_res.items()):#polygons为一个大图内的多边形结果,coco格式
        processed_id = set()
        add_ids = [idx for idx,p in enumerate(polygons) if 'add' in p]  # 筛选出 'add' 标记的多边形
        
        # 使用 STRtree 进行空间索引加速
        polygon_list = [p['polygon'] for p in polygons]
        str_tree = STRtree(polygon_list)
        
        for add_id in add_ids:# 遍历每个 add 多边形
            add_polygon = polygons[add_id]['polygon']
            
            # 使用空间索引来查找可能与 add_polygon 相交的多边形
            possible_intersections = str_tree.query(add_polygon)
            
            for other_id in possible_intersections:# 仅处理在空间索引中查找到的多边形
                if other_id == add_id or other_id in processed_id:
                    continue# 跳过自身以及已处理的多边形
                other_item = polygons[other_id]
                other_polygon = other_item['polygon']
                # 判断是否存在实际交集
                if add_polygon.intersects(other_polygon):
                    # 对两个多边形都应用缓冲，可填补中间的buffer_distance/2的小缝隙，用add_polygon的边界
                    # eroded_add_polygon = add_polygon.buffer(-buffer_distance/2)
                    # expanded_other_polygon = other_polygon.buffer(buffer_distance)
                    cutted_polygon = other_polygon.difference(add_polygon)
                    # cutted_polygon = expanded_other_polygon.difference(eroded_add_polygon).buffer(-buffer_distance)
                    
                    # 更新 other_item 的 segmentation
                    if not cutted_polygon.is_empty:
                        if cutted_polygon.geom_type in ['MultiPolygon', 'GeometryCollection']:
                            for i, poly in enumerate(cutted_polygon.geoms):
                                if poly.geom_type == 'LineString':
                                    continue
                                if poly.area<area_low_thr:#[other_item['category_id']]:
                                    continue
                                if i == 0:#第一个多边形更新到原有other_item
                                    other_item['segmentation'] = polygon2cocoSeg(poly)
                                    other_item['polygon'] = poly
                                else:#其余多边形按coco格式保存到new_merge_res
                                    new_item = other_item.copy()
                                    new_item['segmentation'] = polygon2cocoSeg(poly)
                                    new_item.pop('polygon')
                                    new_item['id'] = n
                                    minx, miny, maxx, maxy = poly.bounds
                                    new_item['bbox'] = (minx, miny, maxx - minx, maxy - miny)
                                    new_item['area'] = poly.area
                                    n+=1#id重新编号
                                    new_merge_res.append(new_item)  # 添加新多边形到结果中
                        else:#单个多边形
                            if cutted_polygon.area<area_low_thr:#[other_item['category_id']]:
                                other_item['segmentation'] = []#小于阈值的多边形不要
                                # print('small area')
                                continue
                            other_item['segmentation'] = polygon2cocoSeg(cutted_polygon)
                            other_item['polygon'] = cutted_polygon  # 更新为新的 Polygon 对象
                    else:
                        other_item['segmentation'] = []  # 如果差集为空，移除该多边形
                #由于other_item可能和多个add多边形相交，所以更新other_item后还不能加入processed_id
            # 将 add_id 加入已处理集合
            processed_id.add(add_id)

    for image_id, items in grouped_res.items():
        for item in items:
            item.pop('polygon')
            if len(item['segmentation']) == 0:
                continue
            new_merge_res.append(item)
    
    return new_merge_res

if __name__ == '__main__':
    args=parse_args()
    merge_res=merge_result(args.res_seg,args.res_prompt_s,args.res_prompt_l)
    # 将 merge_res 按照 image_id 分组
    grouped_res = group_by_image_id(merge_res)

    # 处理重叠问题
    final_merge_res = handle_polygon_overlap(grouped_res,len(merge_res))
    with open(args.out,'w') as f:
        json.dump(final_merge_res,f)
    print(f"Merge result saved to {args.out}")
    print(len(merge_res),len(final_merge_res))
    