import json
from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import shape
from rasterio import features
import sys,os
sys.path.append(os.path.abspath('./'))#add project root to PATH
from dataset.data_utils import polygon2cocoSeg,cocoSeg2polygon
from shapely.strtree import STRtree

# 提取特定类别的多边形，并进行简化
def extract_polygons(seg, values_to_extract, simplify_tolerance=4):
    extracted_polygons = {} #[category_id: [polygons(Shapely Polygon),...]]
    
    for category_id in set(values_to_extract).intersection(np.unique(seg)):
        mask = seg == category_id

        for region, category_id in list(features.shapes(seg, mask=mask, connectivity=8)):
            geom = shape(region)# 转换成Shapely几何对象
            geom = geom.simplify(simplify_tolerance)# DP算法简化几何对象
            # if not geom.is_valid:
            #     geom=geom.buffer(0)
            if category_id in extracted_polygons:
                extracted_polygons[category_id].append(geom)
            else:
                extracted_polygons[category_id] = [geom]
    return extracted_polygons

# 计算IoU
def calculate_iou(poly1, poly2):
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union

# 处理每个图像的标注
def process_image_annotations(coco, image_id, seg_pred, values_to_extract,match_n, Tmp=0.5, simplify_tolerance=1.0):
    anns_new=[]
    match_n,total_n=match_n
    extracted_polygons = extract_polygons(seg_pred, values_to_extract, simplify_tolerance)
    for category_id in values_to_extract:
        if category_id not in extracted_polygons:
            continue        
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=[category_id]))
        preds=extracted_polygons[category_id]
        preds_tree=STRtree(preds)
        total_n+=len(anns)
        for ann in anns:
            if len(ann['segmentation'][0])<8:#小于三个点
                continue
            annotation_poly =cocoSeg2polygon(ann['segmentation'])
            for pred_id in preds_tree.query(annotation_poly):#只从匹配的preds中找
                pred=preds[pred_id]
                if not pred.is_valid:
                    continue
                iou = calculate_iou(annotation_poly, pred)
                if iou > Tmp:
                    match_n+=1
                    polygon=polygon2cocoSeg(pred)
                    ann['mask_prompt']=polygon['segmentation']
                    break
            anns_new.append(ann)
    return anns_new,(match_n,total_n)
def process(ann_file, segmentation_dir, gt_dir, save_ann_file,values_to_extract,gt_cls=2, Tmp=0.7, simplify_tolerance=2):
    coco = COCO(ann_file)
    image_ids = coco.getImgIds()

    ann_data=json.load(open(ann_file))
    anns=[]
    match_n=(0,0)#匹配数，总数

    for i,image_id in enumerate(tqdm(image_ids)):
        # if i>100:
        #     break
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        segmentation_file = f"{segmentation_dir}/{file_name}"
        
        seg_pred = cv2.imread(segmentation_file, cv2.IMREAD_GRAYSCALE)
        seg_gt = cv2.imread(f"{gt_dir}/{file_name}", cv2.IMREAD_GRAYSCALE)
        #gt道路类别赋值给pred:
        seg_pred[seg_gt==gt_cls]=gt_cls
        updated_annotations,match_n = process_image_annotations(coco, image_id, seg_pred, values_to_extract,match_n, Tmp, simplify_tolerance)
        anns.extend(updated_annotations)
    print('匹配比例：',match_n[0]/match_n[1])
    # 保存更新后的标注
    ann_data['annotations'] = anns
    with open(save_ann_file, 'w') as f:
        json.dump(ann_data, f)
    return

"""
遍历每个ann_small, large的gt多边形，匹配分割结果（不用顶点数阈值）
将语义分割结果栅格图提取独立多边形，(匹配IoU阈值0.7)作为mask_prompt，
添加到矢量标注文件中
""" 
simplify_tolerance=1
#loveda:
for mode in ['Val','Train']:
    for t in ['large','small']:       
        Tmp=0.85 if t=='large' else 0.7  #标注和语义分割预测实例多边形的IoU阈值
        ann_file = f'dataset/loveda/{mode}/Rural/ann_{t}.json'
        segmentation_dir = f'/irsa/irsa_wch/SegSAMPoly/mmsegmentation/workdir_loveda/mask2former_lda/{mode.lower()}'#val Rural train 无Rural
        gt_dir=f'dataset/loveda/{mode}/Rural/masks_png/'
        save_ann_file = ann_file.replace('.json', '_prompt.json')
        values_to_extract = [1]#建筑类
        gt_cls=2
        process(ann_file, segmentation_dir, gt_dir, save_ann_file,values_to_extract,gt_cls, Tmp, simplify_tolerance)
