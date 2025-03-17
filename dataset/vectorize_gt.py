import json
from shapely.geometry import shape
from rasterio import features
import cv2
import os
from os.path import join
from tqdm import tqdm
from dataset.data_utils import polygon2cocoSeg
# 提取特定类别的多边形，并进行简化
def extract_polygons(seg, simplify_tolerance=4):
    extracted_polygons = {} #[category_id: [polygons(Shapely Polygon),...]]
    cls2extract=[1]
    for category_id in cls2extract:#np.unique(seg):
        mask = seg == category_id

        for region, category_id in list(features.shapes(seg, mask=mask, connectivity=8)):
            geom = shape(region)# 转换成Shapely几何对象
            geom = geom.simplify(simplify_tolerance)# DP算法简化几何对象
            if category_id in extracted_polygons:
                extracted_polygons[category_id].append(geom)
            else:
                extracted_polygons[category_id] = [geom]
    return extracted_polygons


def process_seg(image_id, seg,Ts=160, simplify_tolerance=4):
    extracted_polygons = extract_polygons(seg, simplify_tolerance) 
    all,large,small = [],[],[]
    for category_id, polygons in extracted_polygons.items():
        for poly in polygons:
            area = poly.area
            polygon=polygon2cocoSeg(poly)
            bbox=polygon['bbox']
            ann = {
                'image_id': image_id,
                'category_id': int(category_id),
                'segmentation': polygon['segmentation'],
                "area": round(area,2), 
                "bbox": [round(b, 2) for b in bbox],
                "iscrowd": 0,
                # 'mask_prompt': polygon['segmentation']
            }
            all.append(ann)            
            if bbox[2]>Ts or bbox[3]>Ts:
                large.append(ann)
            else:
                small.append(ann)
    return all,large,small

if __name__ == '__main__':
    Ts=160
    simplify_tolerance=4
    mode='Val'
    #seg_dir下栅格标注保存为coco格式矢量标注，保存为ann.json，ann_large.json, ann_small.json
    save_ann_file = f'dataset/loveda/{mode}/Rural/ann.json'
    seg_dir = f'dataset/loveda/{mode}/Rural/masks_png'

    anns=[]
    alls,larges,smalls=[],[],[]
    match_n=0
    ann_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "building"},
            ]
    }
    files = os.listdir(seg_dir)

    files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i,seg_file in enumerate(tqdm(files)):
        seg = cv2.imread(join(seg_dir,seg_file), cv2.IMREAD_GRAYSCALE)
        all,large,small = process_seg(i, seg,Ts, simplify_tolerance)
        ann_data['images'].append({"id": i, "file_name": seg_file, "width": seg.shape[1], "height": seg.shape[0]})
        alls.extend(all)
        larges.extend(large)
        smalls.extend(small)

    for i,res in enumerate(alls):
        res['id']=i #smalls large也会相应编号
    ann_data['annotations'] = alls
    with open(save_ann_file, 'w') as f:
        json.dump(ann_data, f)
    ann_data['annotations'] = smalls
    with open(save_ann_file.replace('ann','ann_small'), 'w') as f:
        json.dump(ann_data, f)
    ann_data['annotations'] = larges
    with open(save_ann_file.replace('ann','ann_large'), 'w') as f:
        json.dump(ann_data, f)
    print(f'{mode} all:{len(alls)},large:{len(larges)}, small:{len(smalls)}')