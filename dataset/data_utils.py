import numpy as np
import torch
import cv2
from numpy.random import randint
from shapely.geometry import Polygon, box,GeometryCollection,MultiLineString,MultiPolygon,LineString
# from pycocotools import mask as maskUtils

from scipy.ndimage.morphology import distance_transform_edt as distrans
def MarkovNoise(gt, T=10, theta1=0.5, theta2=0.3):
    '''
        Generate the proposed Markov noise
        gt: ndarray, groundtruth mask 0/1
        T: int, iteration number
        theta1: float, probability of flipping the pixel outside(otherwise inside) the mask 
        值越大，mask越容易往外扩展
        theta2: float, probability of flipping the pixel
        
    '''
    noise = gt.copy()
    ps = np.random.rand(T)
    for t in range(T):
        if ps[t] < theta1:
            edge = (distrans(1-noise)==1).nonzero()#距离mask边缘1像素的位置（mask外）的坐标
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
        else:
            edge = (distrans(noise)==1).nonzero()#距离mask边缘1像素的位置（mask内）的坐标
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
    noise = (noise).astype(np.uint8)
    noise = cv2.GaussianBlur(noise, (5, 5), 2, 2)#kernel size, sigmaX, sigmaY
    noise = np.array(noise>0.5)
    # print('Dice {:.2f}'.format(_dice(gt, noise)))
    return noise

def polygon2cocoSeg(polygon: Polygon,cal_num_points=False):
    """
    将shapely Polygon对象转换为COCO格式的segmentation，并返回点数和bbox。
    
    Args:
    polygon (Polygon): shapely的Polygon实例。

    Returns:
    dict: 包含 'segmentation' [[外部轮廓坐标序列],[内部孔洞1],[内部孔洞2]...], 'num_points' (int), 'bbox' (tuple: x, y, w, h)
    """
    # if not polygon.is_valid:
    #     polygon=make_valid(polygon)#会导致多边形变成MultiPolygon,GeometryCollection 下面需要.geom
    # 获取外边界的坐标，确保按顺时针顺序排列
    exterior_coords = list(polygon.exterior.coords)
    # 获取孔洞的坐标，按逆时针顺序排列
    interior_coords = [list(interior.coords) for interior in polygon.interiors]
    # 将坐标平铺到一维数组
    segmentation = [list(sum(exterior_coords, ()))]
    for interior in interior_coords:
        segmentation.append(list(sum(interior, ())))
    # 计算bbox (x_min, y_min, width, height)
    minx, miny, maxx, maxy = polygon.bounds
    bbox = (minx, miny, maxx - minx, maxy - miny)
    res = {
        "segmentation": segmentation,
        "bbox": bbox
    }
    if cal_num_points:
        # 点数是外边界加上所有孔洞的点数
        num_points = len(exterior_coords) + sum(len(interior) for interior in interior_coords)        
        res["num_points"] = num_points
    return res
# 将 COCO segmentation 转换为 shapely Polygon 对象
def cocoSeg2polygon(segmentation):
    """
    将COCO格式的segmentation转换为带有内部孔洞的shapely Polygon对象。
    
    segmentation: [[外部轮廓坐标序列], [内部孔洞1], [内部孔洞2], ...]
    """
    # 第一个序列是外部轮廓
    exterior = list(zip(segmentation[0][::2], segmentation[0][1::2]))
    # 后续的序列（如果有）是内部孔洞
    interiors = [list(zip(hole[::2], hole[1::2])) for hole in segmentation[1:]] if len(segmentation) > 1 else []
    polygon=Polygon(exterior, interiors)# 创建带孔的Polygon对象
    if not polygon.is_valid:
        polygon=polygon.buffer(0)
    return polygon
def get_bbox_point(ann, scale_rate=1, crop_bbox=None, hflip=False, vflip=False, jitter=True,
                   add_center=True,input_size=1024):
    # 几何变换：随机缩放，随机裁剪到crop_bbox，最终垂直/水平翻转，相应的bbox和point也要变换
    # crop_bbox (x1, y1, x2, y2) 左上右下
    bbox = np.array(ann['bbox'])  # x, y, w, h
    #transform:
    if scale_rate != 1:
        bbox *= scale_rate
    w,h=bbox[2],bbox[3]
    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox #x2=x1+512,y2=y1+512
        # 保留bbox在crop_bbox内的部分，重新计算bbox在crop_bbox内的坐标:
        bbox[0] = max(bbox[0], x1)-x1
        bbox[1] = max(bbox[1], y1)-y1
        bbox[2] = min(w, 512-bbox[0])
        bbox[3] = min(h, 512-bbox[1])    
    if hflip:
        bbox[0]=512-bbox[0]-bbox[2] #注意bbox[0]还是左上点
    if vflip:
        bbox[1]=512-bbox[1]-bbox[3]
    #取新的bbox中心点:
    if add_center:
        point = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    if jitter:
        # bbox坐标和point坐标都要加上随机扰动，扰动范围为bbox的w和h的1/10 w,h扰动范围1/20
        jitter_amount_bbox = np.array([bbox[2] / 10, bbox[3] / 10, bbox[2] / 20, bbox[3] / 20])
        bbox += np.random.uniform(-jitter_amount_bbox, jitter_amount_bbox)        
        if add_center:
            jitter_amount_point = np.array([bbox[2] / 10, bbox[3] / 10])            
            point += np.random.uniform(-jitter_amount_point, jitter_amount_point)
    #x,y,w,h->x1,y1,x2,y2
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    bbox=np.clip(bbox,0,512-1e-4)#防止bbox出界
    if input_size!=512:
        bbox*=input_size/512
        #512->input_size
    if add_center:
        if input_size!=512:
            point*=input_size/512
        point=point.reshape(1,2)
        return bbox, point
    else:
        return bbox
def get_prompt_points(polygon,gt_size=256,input_size=1024):
    #随机生成多边形内的1至3个点坐标，且坐标在多边形内靠近边缘的位置的概率更大
    #polygons: np.array(n,2) x,y (w,h)    gt_w,gt_h（gt_size）范围内
    #返回在input_size*input_size(输入sam模型）范围内的坐标
    # Compute polygon centroid
    # centroid = np.mean(polygon[:-1], axis=0)#去掉最后一个点，因为和第一个点重复
    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])
    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])    
    # 计算外接矩形的中心点
    centroid = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    pos_num = 1#np.random.randint(1, 3)#随机生成1-2个点
    # neg_num = np.random.randint(0, 3)#随机生成0-2个负样本点 等概率
    neg_num=np.random.choice([0, 1], p=[0.8, 0.2])#随机生成0或1个负样本点 负样本点生成概率为0.2
    keypoints = [generate_point_inside_polygon(polygon,centroid,l=0.5, h=1) for _ in range(pos_num)]
    
    #坐标变到input_size*input_size范围内:
    keypoints=np.array(keypoints)*input_size/gt_size
    if neg_num>0:
        neg_keypoints=[]
        for _ in range(neg_num):
            neg=generate_point_outside_polygon(polygon,centroid,gt_size=gt_size)
            if neg is None:
                neg_num-=1
            else:
                neg_keypoints.append(neg)
    max_point_num=2
    #关键点列表padding，加标签:
    points=np.zeros((max_point_num,2))#统一补齐成max_point_num个点 无效点用0填充
    points[:pos_num,:]=keypoints
    label=-np.ones(max_point_num)#无效标签用-1填充
    label[:pos_num]=np.ones(pos_num)#正样本标签用1填充
    if neg_num>0:
        neg_keypoints=np.array(neg_keypoints)*input_size/gt_size
        points[pos_num:pos_num+neg_num,:]=neg_keypoints
        label[pos_num:pos_num+neg_num]=np.zeros(neg_num)#负样本标签用0填充
    return points,label

import matplotlib.path as mpltPath
def is_point_inside_polygon(polygon, point):#todo 检查孔洞多边形
    """Check if point is inside polygon."""
    path = mpltPath.Path(polygon)
    return path.contains_point(point)
def random_point_on_edge(polygon):
    """Generate a random point on a random edge of the polygon."""
    # Choose a random edge
    edge_index = np.random.randint(0, len(polygon) - 1)
    p1, p2 = polygon[edge_index], polygon[edge_index + 1]

    # Generate a random weight
    alpha = np.random.random()

    # Interpolate between the two points of the edge
    return [(1 - alpha) * p1[0] + alpha * p2[0], (1 - alpha) * p1[1] + alpha * p2[1]]

def generate_point_inside_polygon(polygon,centroid,l=0.07,h=0.5):
    """
    l=0.07,h=0.5: Generate a point near the edge of the polygon.
    l=0.5, h=1: Generate a point near the centroid of the polygon.
    """
    i=0
    while True:
        # Get random point on edge
        point = random_point_on_edge(polygon)

        # Move the point slightly towards the centroid
        move_ratio = np.random.uniform(l, h)  # 靠近中心点的程度，中心点与边缘点距离为幅度1
        point = [(1 - move_ratio) * point[0] + move_ratio * centroid[0],
                 (1 - move_ratio) * point[1] + move_ratio * centroid[1]]

        # If point is inside the polygon, return it
        if is_point_inside_polygon(polygon, point):
            return point
        i+=1
        if i>3:#防止多次不在多边形内
            return centroid
def generate_point_outside_polygon(polygon,centroid,gt_size=256):
    """Generate a point outside the polygon along the line from centroid to a point on the edge."""
    # Get random point on edge   todo:增加超过图像边界判断 gt_size为边界
    i=0
    while True:
        point = random_point_on_edge(polygon)
        # Move the point beyond the edge, along the line connecting the centroid to the point on the edge
        move_ratio = np.random.uniform(0.1, 0.5)  #沿着中心点到边缘点的线向外移动的程度,中心点与边缘点距离为幅度1 （0.1，0.8）
        point = [(1 + move_ratio) * point[0] - move_ratio * centroid[0],
                (1 + move_ratio) * point[1] - move_ratio * centroid[1]]
        image_boundary=[(0,0),(0,gt_size),(gt_size,gt_size),(gt_size,0)]
        if is_point_inside_polygon(image_boundary, point):
            return point
        i+=1
        if i>3:
            return None

def get_point_ann(junctions,gt_h,gt_w,gaussian=True,sigma_x=1, sigma_y=1):
    #junctions: np.array(n,2) x,y (w,h) gt_w,gt_h(256*256)范围内 vmap, voff大小为gt_h,gt_w
    # x,y方向的高斯滤波核的标准差sigma_x,sigma_y
    #获取顶点激活图vmap和偏移量图voff：
    vmap = np.zeros((gt_h, gt_w))
    # voff = np.zeros((2, gt_h, gt_w))
    junctions[:,0]=np.clip(junctions[:,0],0,gt_w-1e-4)
    junctions[:,1]=np.clip(junctions[:,1],0,gt_h-1e-4)
    xint, yint = junctions[:,0].astype(int), junctions[:,1].astype(int)#向下取整
    # off_x = junctions[:,0] - xint#0~1
    # off_y = junctions[:,1] - yint
    if gaussian:
        # 计算高斯滤波核的大小
        size_x = round(sigma_x * 3)# 3 sigma rule
        size_y = round(sigma_y * 3)
        # 确保 size_x 和 size_y 都为奇数
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        half_size_x = size_x // 2
        half_size_y = size_y // 2    
        vmap = np.zeros((gt_h, gt_w), dtype=np.float32)
        for x, y in zip(xint, yint):
            for i in range(max(0, x - half_size_x), min(gt_w, x + half_size_x + 1)):
                for j in range(max(0, y - half_size_y), min(gt_h, y + half_size_y + 1)):
                    dist_x_square = (i - x) ** 2
                    dist_y_square = (j - y) ** 2
                    value = np.exp(-0.5 * (dist_x_square / sigma_x ** 2 + dist_y_square / sigma_y ** 2))
                    vmap[j, i] = max(vmap[j, i], value)
    else:
        vmap[yint, xint] = 1
    # voff[0, yint, xint] = off_x
    # voff[1, yint, xint] = off_y
    if gaussian:
        return torch.tensor(vmap).float().unsqueeze(0)
    else:
        return torch.tensor(vmap).long().unsqueeze(0)
def get_mask(junctions,seg_h,seg_w):
    #junctions: np.array(n,2) x,y (w,h) (seg_h,seg_w)范围内
    # junctions=[junctions.reshape(-1).tolist()]#[[x1,y1],[x2,y2],...] -> [[x1,y1,x2,y2,...]]
    #junctions coco标注'segmentation'字段
    #由多边形顶点列表获取gt_mask, 大小为seg_h,seg_w；
    #获取分割图gt_mask：
    # rle = maskUtils.frPyObjects(junctions, seg_h,seg_w)#h,w
    # mask = maskUtils.decode(rle)
    mask = np.zeros((seg_h, seg_w), dtype=np.uint8)
    cv2.fillPoly(mask, [junctions.astype(np.int32)], color=1)
    return mask
def generate_edge_map(junctions, gt_h, gt_w,gaussian=True,sigma_x=1, sigma_y=1):
    #junctions: np.array(n,2) x,y (w,h) (gt_h,gt_w)范围内
    junctions = junctions.astype(np.int32)
    edge_map = np.zeros((gt_h, gt_w), dtype=np.uint8)
    edge_map = cv2.polylines(edge_map, [junctions], isClosed=True, color=1, thickness=1)
    if gaussian:
        edge_map=edge_map.astype(np.float32)
        size_x = round(sigma_x * 3)# 3 sigma rule
        size_y = round(sigma_y * 3)
        # 确保 size_x 和 size_y 都为奇数
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        gaussian_map = cv2.GaussianBlur(edge_map,(size_x, size_y), sigmaX=sigma_x, sigmaY=sigma_y)
        edge_map = gaussian_map / np.max(gaussian_map)
    return edge_map
def clip_polygon_by_bbox(polygon_points: np.ndarray, bbox: tuple):
    """裁剪多边形与边界框的交集部分。    
    Args:
        polygon_points (np.ndarray): 多边形顶点的坐标，形状为(n,2)。
        bbox (tuple): 边界框的四个顶点坐标，格式为(minx, miny, maxx, maxy)。
    Returns:
        np.ndarray: 裁剪后的多边形顶点的坐标。(n,2)
    """
    # 创建Polygon和Box对象
    polygon = Polygon(polygon_points)
    bbox_polygon = box(*bbox)
    # 计算交集
    intersection_polygon = polygon.intersection(bbox_polygon)
    # 如果交集为多边形，则返回其顶点坐标
    if intersection_polygon.is_empty:
        return np.array([])
    else:
        # shapely.coords.CoordinateSequence
        if isinstance(intersection_polygon,(GeometryCollection,MultiPolygon)):
            for geometry in intersection_polygon.geoms:
                if isinstance(geometry, Polygon):
                    # 如果是多边形类型，则执行相应操作
                    polygon = geometry
                    return np.array(polygon.exterior.coords)
        elif isinstance(intersection_polygon,(MultiLineString,LineString)):
            return np.array([])
        else:#Polygon
            return np.array(intersection_polygon.exterior.coords)
        #todo AttributeError: 'Point' object has no attribute 'exterior'
