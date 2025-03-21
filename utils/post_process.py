from shapely.geometry import Polygon
from .polygon import get_candidate_vertex,mask_guided_vertex_connect
from skimage.measure import label, regionprops
import numpy as np
from pycocotools import mask as coco_mask
import cv2
def fill_holes(mask, fill_all=False,thr=0.01):
    """
    Fill the holes in the input binary mask.
    - mask: numpy ndarray, binary image
    Returns:
    - mask_filled: numpy ndarray, binary image with holes filled
    """
    if fill_all:
        #仅检索最外层的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 填补找到的连通组件
        cv2.drawContours(mask, contours, -1, (1), thickness=cv2.FILLED)#直接在mask上修改
    else:#只填补面积小于阈值的：
        original_area = np.sum(mask == 1)
        # 查找所有连通组件
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            # 如果轮廓面积小于原始 mask 面积的 thr 倍，则填补
            if area < original_area * thr:
                cv2.drawContours(mask, [contour], -1, (1), thickness=cv2.FILLED)
def process_single_instance(args):
    b, pred_seg, candidate_vertex, scale_size, max_distance, pos_transform = args

    scale_y, scale_x = scale_size
    if pos_transform is not None:
        scale_x *= pos_transform[2]#roi区域/原图尺寸(224)
        scale_y *= pos_transform[3]

    candidate_vertex[:, 0] *= float(scale_x)#特征图坐标转换为原图坐标 特征图坐标*（原图/特征图）尺度因子（*（roi区域/原图）尺度因子）
    candidate_vertex[:, 1] *= float(scale_y)

    if pos_transform is not None:
        #重采样到原始图片中的尺寸：（之前已经resize到原图大小）
            #若pred_segs[b]resize后小于2*2则不进行resize
        if pred_seg.shape[1] * pos_transform[2] < 2 or pred_seg.shape[0] * pos_transform[3] < 2:
            return None
        pred_seg = cv2.resize(pred_seg, dsize=None, fx=pos_transform[2], fy=pos_transform[3])
    else:
        pred_seg = pred_seg

    pred_seg_binary = (pred_seg > 0.5)
    labels = label(pred_seg_binary)
    props = regionprops(labels)

    if len(props) > 1:
        props = sorted(props, key=lambda x: x.area, reverse=True)

    if props:
        # if len(props) > 1 and props[0].area / props[1].area < 8:#第二大部分面积大于第一大部分的1/8
        #     props = props[:2]
        # else:
        #     props = [props[0]]
        props = [props[0]]
    else:
        return None

    mask_shape = pred_seg.shape
    filt_mask = np.zeros(mask_shape).astype(np.uint8)
    for i, prop in enumerate(props):
        coords = prop.coords
        if i == 0:
            best_score = np.mean(pred_seg[coords[:, 0], coords[:, 1]])#实例mask的平均概率
        else:
            best_score = (best_score + np.mean(pred_seg[coords[:, 0], coords[:, 1]])) / 2
        filt_mask[coords[:, 0], coords[:, 1]] = 1

    fill_holes(filt_mask)#todo regionprops和fill_holes换为先腐蚀去掉噪点，再膨胀填充孔洞

    if isinstance(max_distance, list):
        max_dis = max_distance[b]
    else:
        max_dis = max_distance

    poly,low_iou_num = mask_guided_vertex_connect(filt_mask, candidate_vertex, max_distance=max_dis)
    if poly.shape[0] < 3:
        return None
    
    if pos_transform is not None:
        poly=transform_polygon_to_original(poly,pos_transform,allready_scale=True)
    return poly, best_score, b,low_iou_num

def GetPolygons(pred_segs, pred_vmaps, scale_size,
                 max_distance=12, pos_transforms=None,pool=None):
    """
    input: 模型预测的结果
        pred_segs: (b,h,w)含b个实例的多边形分割概率图(torch.sigmoid().cpu().numpy()结果) 已经resize到ori_size
        pred_vmaps: (b,1,h,w) 特征图大小
        pred_voffs: (b,2,h,w)
        ori_size:原始图片大小
        max_distance: max distance(T_dist in paper) to retain vertex in mask guided vertex connection. int or list(b)
        pos_transforms: (b,4) [x,y,scale_x,scale_y] x,y为左上角坐标，scale_x,scale_y为x,y轴的尺度因子
            pos_transforms非空时，将预测的多边形坐标尺寸转换为原始图片尺寸
    output:
        batch_polygons: [ndarray(n, 2),...] b个 (x,y)对应width, height
        batch_scores: ndarray(b) 各实例的mask score
        valid_mask: ndarray(b) True/False 有效性标记
    """
    num_instances = pred_segs.shape[0]
    batch_polygons = [None] * num_instances  # 初始化包含None的列表，由于点数不统一，无法用np.array
    batch_scores = np.zeros(num_instances,dtype=np.float32)
    valid_mask = np.zeros(num_instances, dtype=bool)  # 初始化有效性标记为False
    low_iou_Num=0
    candidate_vertices = [get_candidate_vertex(pred_vmaps[b]) for b in range(num_instances)]
    # candidate_vertices =get_candidate_vertex(pred_vmaps, pred_voffs)
    scale_size=(scale_size,scale_size)#原图/特征图尺寸=4 crop输入时原图大小为224
    args = [
        (
            b, pred_segs[b], candidate_vertices[b],
            scale_size, max_distance,
            pos_transforms[b] if pos_transforms is not None else None
        )
        for b in range(num_instances)
    ]
    if pool is not None:
        results = pool.map(process_single_instance, args)#imap_unordered map
    else:#single process
        results = map(process_single_instance, args)
    for result in results:        
        if result is not None:
            poly, best_score, idx,low_iou_num = result #idx限定了位置，imap_unordered，不会改变顺序
            batch_polygons[idx] = poly
            batch_scores[idx] = best_score
            valid_mask[idx] = True  
            low_iou_Num+=low_iou_num

    return batch_polygons, batch_scores, valid_mask,candidate_vertices,low_iou_Num
def GetPolygons_s(pred_segs, pred_vmaps, pred_voffs,ori_size=(512,512),max_distance=12,pos_transforms=None):
    
    batch_polygons = []
    batch_scores = []
    valid_idx = []
    two=[]
    no_mask=0
    for b in range(pred_segs.shape[0]):
        candidate_vertex = get_candidate_vertex(pred_vmaps[b], pred_voffs[b])
        scale_y, scale_x = ori_size[1]/pred_vmaps.shape[2], ori_size[0]/pred_vmaps.shape[3]#原图/特征图尺寸=4 crop输入时原图大小为224
        if pos_transforms is not None:
            scale_x *= pos_transforms[b][2]#roi区域/原图尺寸(224)
            scale_y *= pos_transforms[b][3]
        candidate_vertex[:,0] *= float(scale_x)#特征图坐标转换为原图坐标 特征图坐标*（原图/特征图）尺度因子（*（roi区域/原图）尺度因子）
        candidate_vertex[:,1] *= float(scale_y)
        if pos_transforms is not None:
            #重采样到原始图片中的尺寸：（之前已经resize到原图大小）
            #若pred_segs[b]resize后小于2*2则不进行resize
            if pred_segs[b].shape[1]*pos_transforms[b][2]<2 or pred_segs[b].shape[0]*pos_transforms[b][3]<2:
                no_mask+=1
                continue
            pred_seg=cv2.resize(pred_segs[b],dsize=None,fx=pos_transforms[b][2],fy=pos_transforms[b][3])           
        else:
            pred_seg=pred_segs[b]
        pred_seg_binary=(pred_seg>0.5)
        labels = label(pred_seg_binary)
        props = regionprops(labels)
        
        if len(props) >1:
            props = sorted(props, key=lambda x: x.area, reverse=True) #选择面积最大的props 实际效果与置信度最高的一致
        if props:
            props = [props[0]]
            # if len(props) >1:
            #     # with open('/irsa/irsa_wch/SAMPolyBuild/work_dir/loss_weight/iou50/test1/multi_part_area.txt','a+') as f:
            #     #     for prop in props:
            #     #         f.write(str(int(prop.area))+', ')
            #     #     f.write('\n')
            #     if props[0].area/props[1].area<8:#第二大部分面积大于第一大部分的1/8
            #         two.append(b)
            #         props=props[:2]#取前两个
            # else:
            #     props = [props[0]]
            # continue# to delete
        else: 
            no_mask+=1#pred_seg_binary全0 重采样后长条一边为1情况
            continue
        mask_shape=pred_seg.shape
        filt_mask = np.zeros(mask_shape).astype(np.uint8)
        for i,prop in enumerate(props):
            coords = prop.coords  # 获取该 prop 的所有坐标点
            if i == 0:
                best_score = np.mean(pred_seg[coords[:, 0], coords[:, 1]])#实例mask的平均概率
            else:
                best_score = (best_score + np.mean(pred_seg[coords[:, 0], coords[:, 1]])) / 2
            filt_mask[coords[:, 0], coords[:, 1]] = 1
        fill_holes(filt_mask) 
        if isinstance(max_distance,list):
            max_dis=max_distance[b]
        else:
            max_dis=max_distance
        poly,low_iou_num=mask_guided_vertex_connect(filt_mask, candidate_vertex,max_distance=max_dis)
        if poly.shape[0] <3:
            no_mask+=1
            continue
        if pos_transforms is not None:
            poly=transform_polygon_to_original(poly,pos_transforms[b],allready_scale=True)
        valid_idx.append(b)
        batch_scores.append(best_score)
        batch_polygons.append(poly)
    return batch_polygons, batch_scores,valid_idx#,no_mask#,two

def poly2bbox(poly):
    """
    input: (n, 2)
    output: (4)
    """
    x_min = np.min(poly[:,0])
    x_max = np.max(poly[:,0])
    y_min = np.min(poly[:,1])
    y_max = np.max(poly[:,1])
    return np.array([x_min, y_min, x_max-x_min, y_max-y_min]).tolist()
def transform_coords_to_original(x, y, pos_transform,allready_scale):
    """
    Convert coordinates from the cropped image to the original image.
    pos_transform:[x,y,scale_x,scale_y] x,y为roi区域左上角在原图的坐标，scale_x,scale_y为x,y轴的尺度因子 scale_x=原图roi宽/统一roi宽
    """
    if allready_scale:
        orig_x = x + pos_transform[0]
        orig_y = y + pos_transform[1]
    else:
        orig_x = x * pos_transform[2] + pos_transform[0]
        orig_y = y * pos_transform[3] + pos_transform[1]
    return orig_x, orig_y
def transform_polygon_to_original(polygon, pos_transform,allready_scale=False):
    #polygon: (n, 2)
    orig_polygon = []
    for i in range(len(polygon)):
        x, y = polygon[i]
        orig_x, orig_y = transform_coords_to_original(x, y, pos_transform,allready_scale)
        orig_polygon.append([orig_x, orig_y])
    return np.array(orig_polygon)
def original2crop(polygon,pos_transform):
    #polygon: (n, 2)
    crop_polygon = []
    for i in range(len(polygon)):
        x, y = polygon[i]
        crop_x, crop_y = (x - pos_transform[0]) / pos_transform[2], (y - pos_transform[1]) / pos_transform[3]
        crop_polygon.append([crop_x, crop_y])
    return np.array(crop_polygon)
def generate_coco_ann(polygon, score, img_id,ann_id=None,category_id=1):
    # polygon: ndarray(n, 2)
    # score: float
    # img_id: int
    # 获取一个建筑实例的COCO格式的标注
    poly_bbox = poly2bbox(polygon)#xywh
    vec_poly = polygon.ravel().tolist()
    ann_per_instance = {
            'image_id': img_id,
            'category_id': category_id,
            'segmentation': [[round(p,1) for p in vec_poly]],
            'area': Polygon(polygon).area,
            'bbox': [round(p,2) for p in poly_bbox],
            'score': float(score),
        }
    if ann_id is not None:
        ann_per_instance['id']=ann_id
    return ann_per_instance
import pycocotools.mask as maskUtils
def generate_coco_mask(mask,score, img_id):
    # mask: ndarray(w, h) 一个建筑物实例的mask
    encoded_region = coco_mask.encode(np.asfortranarray(mask))
    ann_per_building = {
        'image_id': img_id,
        'category_id': 1,
        'segmentation': {
            "size": encoded_region["size"],
            "counts": encoded_region["counts"].decode()
        },
        'score': float(score),
    }
    ann_per_building['area']=int(maskUtils.area(ann_per_building['segmentation']))
    return ann_per_building