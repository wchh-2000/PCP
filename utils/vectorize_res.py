import os,json
from tqdm import tqdm
import cv2
from os.path import join
import argparse
from shapely.geometry import shape
from rasterio import features
import sys
sys.path.append(os.path.abspath('./'))#add cwd
from dataset.data_utils import polygon2cocoSeg
import numpy as np
def extract_polygons(seg, values_to_extract, simplify_tolerance=4):
    extracted_polygons = {} #[category_id: [polygons(Shapely Polygon),...]]
    values=set(np.unique(seg))
    for category_id in values_to_extract:
        if category_id not in values:
            continue
        mask = seg == category_id

        for region, category_id in list(features.shapes(seg, mask=mask, connectivity=4)):
            geom = shape(region)# 转换成Shapely几何对象
            geom = geom.simplify(simplify_tolerance)# DP算法简化几何对象
            if category_id in extracted_polygons:
                extracted_polygons[category_id].append(geom)
            else:
                extracted_polygons[category_id] = [geom]
    return extracted_polygons
def process_seg(image_id, seg,vertex_n_thr,values_to_extract,Ts,
         simplify_tolerance=1.5,min_thr=None):
    """
    values_to_extract: 提取的类别，有顺序
    vertex_n_thr 各类顶点数阈值，大于阈值的多边形不要 eg:{}
    Ts 划分大小实例的bbox边长阈值
    min_thr 最小面积阈值
    'farmland':1, 'garden':2, 'forest':3, 'grassland':4, 'build-up area':5, 'mining land':6, 'road':7, 'water':8, 'bare land':9
    """
    extracted_polygons = extract_polygons(seg,values_to_extract, simplify_tolerance=1) 
    all,large,small = [],[],[]
    for category_id, polygons in extracted_polygons.items():
        for poly in polygons:
            #simp=1的多边形poly_coco用于提示的mask，默认圆滑多边形，simplify_tolerance的poly用于判断点数，存储简单多边形
            area = poly.area
            poly_coco=polygon2cocoSeg(poly)
            bbox=poly_coco['bbox']
            if min_thr is not None and area<min_thr:
                continue
            ann = {
                'image_id': image_id,
                'category_id': int(category_id),
                'segmentation': poly_coco['segmentation'],
                "area": round(area,2), 
                "bbox": [round(b, 2) for b in bbox],
                "iscrowd": 0,
            }
            all.append(ann)
            poly_simp=polygon2cocoSeg(poly.simplify(simplify_tolerance),cal_num_points=True)
            if category_id in vertex_n_thr and poly_simp['num_points'] <=vertex_n_thr[category_id]:#in set([1,3,4])           
                #根据landuse/get_large_ann.py面积阈值
                ann['segmentation']=poly_simp['segmentation']#修改segmentation为简化后的
                ann['mask_prompt']=poly_coco['segmentation']#添加mask_prompt 为simp=1的mask
                if bbox[2]>Ts or bbox[3]>Ts:
                    large.append(ann)
                else:
                    small.append(ann)
    return all,large,small
def vectorize(pred_dir,gt_ann,out_coco_pth,road_cls,vertex_n_thr,values_to_extract,Ts,min_thr, tolerance=4):
    """
    将栅格png转为coco格式.json
    :param preds: 预测的结果文件名列表
    :param tolerance: 控制简化程度的容差。The maximum allowed geometry displacement. The higher this category_id, the smaller the number of vertices in the resulting geometry
    :return:
    """
    with open(gt_ann) as f:
        gt_anns = json.load(f)
    coco_data={"images":gt_anns['images'],"annotations":[],"categories":gt_anns['categories']}
    res_large,res_small,res_all = [],[],[]
    road_n=5#道路膨胀腐蚀次数
    # 定义要按顺序提取的值
    # files = sorted(preds, key=lambda x: int(''.join(filter(str.isdigit, x))))#loveDA
    for img_info in tqdm(gt_anns['images']):#遍历结果图
        pred_name=img_info['file_name']
        i=img_info['id']
        # if i>20:
        #     break
        # pred_name=pred_name.replace('.tif','.png')
        pred=cv2.imread(join(pred_dir,pred_name),cv2.IMREAD_GRAYSCALE)        
        # gt=cv2.imread(join(gt_dir,pred_name),cv2.IMREAD_GRAYSCALE)
        # pred[gt==road_cls]=road_cls#gt道路类别赋值给pred

        mask = (pred == road_cls).astype(np.uint8)
        # 使用膨胀操作 (dilate)，结构元素为 3x3，迭代 4 次
        dilated_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=road_n)
        # 使用腐蚀操作 (erode)，结构元素为 3x3，迭代 4 次
        eroded_mask = cv2.erode(dilated_mask, np.ones((3, 3), np.uint8), iterations=road_n)
        # 将处理后的区域覆盖回 pred
        pred[eroded_mask == 1] = road_cls

        all,large,small = process_seg(i, pred,vertex_n_thr,values_to_extract,Ts, tolerance,min_thr=min_thr)
        
        res_all.extend(all)
        res_large.extend(large)
        res_small.extend(small)    
    print(f"total polygons {len(res_all)}, large polygons {len(res_large)}, small polygons {len(res_small)}")
    for i,res in enumerate(res_all):
        res['id']=i #smalls large也会相应编号
    with open(out_coco_pth, 'w') as f:
        json.dump(res_all, f)
    coco_data["annotations"]=res_large
    large_ann_file=out_coco_pth.replace('.json','_large.json')
    with open(large_ann_file, 'w') as f:
        json.dump(coco_data, f)
    small_ann_file=out_coco_pth.replace('.json','_small.json')
    coco_data["annotations"]=res_small
    with open(small_ann_file, 'w') as f:
        json.dump(coco_data, f)
    print(f"save to {out_coco_pth}")
    print(f"{large_ann_file}, {small_ann_file}")
    return
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_dir', help='prediction directory of semantic segmentation results',
            default='priv/results/loveda/sfanet_val/Rural')
    parser.add_argument(
        '--save_dir', help='directory where coco results will be saved',default='dataset/loveda/Val/Rural/res')
    parser.add_argument(
        '--result_name',default='seg')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args=parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    out_coco_pth=f"{args.save_dir}/{args.result_name}.json"
    #loveda:
    gt_ann='dataset/loveda/Val/Rural/ann.json'
    road_cls=2
    vertex_n_thr={1:500}
    values_to_extract=[1]
    Ts=160
    min_thr=85
    tolerance=8
    vectorize(args.pred_dir,gt_ann,out_coco_pth,road_cls,vertex_n_thr,values_to_extract,Ts,min_thr, tolerance)