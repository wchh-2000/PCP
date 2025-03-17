from pycocotools.coco import COCO #2.0.7
import numpy as np
import json
from tqdm import tqdm
import cv2

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)
    return I, U

def coco_to_mask(segmentation, mask_shape):
    """
    Convert coco format segmentation to mask
    segmentation: [[exterior contour], [hole1], [hole2]...]
    mask_shape: (h,w)
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)
    exterior = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
    # 假设 mask 的尺寸是 (height, width)
    # height, width = mask.shape[:2]

    # # 确保坐标在 [0, width-1] 和 [0, height-1] 范围内
    # exterior[:, 0] = np.clip(exterior[:, 0], 0, width-1)
    # exterior[:, 1] = np.clip(exterior[:, 1], 0, height-1)

    # Fill exterior boundary
    cv2.fillPoly(mask, [exterior], 1)
    if len(segmentation) > 1:  # Has holes
        for hole in segmentation[1:]:
            hole = np.array(hole).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [hole], 0)
    return mask
    
def get_mask_pointNum(coco, img, category_ids): 
    # 获取图像img的mask和所有多边形点数
    annotation_ids = coco.getAnnIds(imgIds=img['id'], catIds=category_ids)
    annotations = coco.loadAnns(annotation_ids)
    mask_shape = (img['height'], img['width'])
    mask = np.zeros(mask_shape, dtype=np.uint8)
    total_points = 0

    for ann in annotations:
        mask += coco_to_mask(ann['segmentation'], mask_shape)
        n_point = sum([len(poly) // 2 for poly in ann['segmentation']])
        # if n_point>200:
        #     continue
        total_points += n_point

    return mask != 0, total_points

def compute_IoU_cIoU(input_json, gti_annotations,value_mapping, category_ids=[1], ignore_index=None):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    if type(submission_file) == dict:
        submission_file = submission_file['annotations']
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)

    image_ids = coco.getImgIds()

    # Initialize variables for overall accumulations
    overall_I = {cat_id: 0 for cat_id in category_ids}
    overall_U = {cat_id: 0 for cat_id in category_ids}
    overall_N_pred = {cat_id: 0 for cat_id in category_ids}
    overall_N_gti = {cat_id: 0 for cat_id in category_ids}
    i=0
    for image_id in tqdm(image_ids):
        # i+=1
        # if i>20:
        #     break
        img = coco.loadImgs(image_id)[0]
        if ignore_index is not None:
            mask_ignore,_ = get_mask_pointNum(coco_gti, img, [ignore_index])
        for category_id in category_ids:
            mask_pred, N_pred = get_mask_pointNum(coco, img, [category_id])
            mask_gti, N_gti = get_mask_pointNum(coco_gti, img, [category_id])
            if ignore_index is not None:
                # Ignore background classes
                valid_mask = ~mask_ignore
                mask_pred = mask_pred[valid_mask]
                mask_gti = mask_gti[valid_mask]

            # Skip this category if both masks have no remaining valid points
            if mask_pred.size == 0 and mask_gti.size == 0:
                continue

            # Accumulate N_pred and N_gti
            overall_N_pred[category_id] += N_pred
            overall_N_gti[category_id] += N_gti

            # Calculate IoU for this category in this image and accumulate I, U
            I, U = calc_IoU(mask_pred, mask_gti)
            overall_I[category_id] += I
            overall_U[category_id] += U

    print("Done!")
    

    print("Overall Mean IoU per class:")
    class_miou_list = []
    class_mciou_list = []
    class_metrics={}
    for cat_id in category_ids:
        if overall_U[cat_id] > 0:
            # Calculate per-class IoU
            miou = overall_I[cat_id] / overall_U[cat_id]
            # Calculate per-class C-IoU
            ps = 1 - np.abs(overall_N_pred[cat_id] - overall_N_gti[cat_id]) / (overall_N_pred[cat_id] + overall_N_gti[cat_id])
            mciou = miou * ps

            # Store per-class results
            class_miou_list.append(miou)
            class_mciou_list.append(mciou)
            miou=round(miou*100, 2)
            mciou=round(mciou*100, 2)
            ps=round(ps*100, 2)
            compare='more' if overall_N_pred[cat_id] > overall_N_gti[cat_id] else 'less'
            class_metrics[cat_id]={'miou':miou,'mciou':mciou,'ps':ps,'compare':compare}
            print(f"{value_mapping[cat_id]}: IoU: {miou:.2f}, C-IoU: {mciou:.2f}, ps: {ps:.2f}%, {compare} pred vertices")

    # Calculate overall mean IoU and C-IoU across all categories
    overall_miou = np.mean(class_miou_list)
    overall_mciou = np.mean(class_mciou_list)
    overall_miou=round(overall_miou*100, 2)
    overall_mciou=round(overall_mciou*100, 2)
    class_metrics['overall']={'miou':overall_miou,'mciou':overall_mciou}

    # Print overall results
    print(f"Overall Mean IoU: {overall_miou:.2f}")
    print(f"Overall Mean C-IoU: {overall_mciou:.2f}")
    return class_metrics

