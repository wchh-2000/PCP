import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from eval.iou import iou_poly

def non_maximum_suppression(a,size=3):#a shape:[1, 256, 256] or [b,1, 256, 256]
    ap = F.max_pool2d(a, size, stride=1, padding=1)#shape:[1, 256, 256]
    if size==2:
        ap=ap[...,:-1,:-1]
    mask = (a == ap).float().clamp(min=0.0)#保留最大值，周围值置0
    return a * mask

def get_candidate_vertex(vmap):#voff_map
    #vmap:(1,h,w),voff_map:(2,h,w)
    pred_nms = non_maximum_suppression(vmap,size=3)
    vertex=pred_nms.squeeze()>0.1#选取大于阈值的点 vertex：true false图
    idx=torch.nonzero(vertex)#选取true的点的坐标 （n,2）
    y=idx[:,0].float()#由于vmap (h,w),所以索引为(y,x)
    x=idx[:,1].float()
    # off=voff_map[:,vertex]
    # dx,dy=off[0],off[1]#voff两通道(x,y)
    # no off:
    # candidate_vertex=torch.stack((x,y)).t()
    candidate_vertex=torch.stack((x,y)).t()#x+dx,y+dy
    return candidate_vertex.detach().cpu().numpy()
def get_candidate_vertex_batch(vmap, scale_size=4):#voff_map,
    # vmap: (b, 1, h, w), voff_map: (b, 2, h, w)    
    # size=3 if large else 2
    # max_k=64 if large else 128
    size=3
    max_k=150#large
    pred_nms = non_maximum_suppression(vmap, size=size)#large3 small2
    vertex_scores = pred_nms.squeeze(1)  # 形状为 (b, h, w)
    vertex_mask = vertex_scores > 0.1  # 形状为 (b, h, w)
    device=vmap.device
    b, h, w = vertex_mask.shape
    max_candidates = vertex_mask.sum(dim=(1, 2)).max().item()  # (b,)->最大候选数量
    k=min(max_k,max_candidates)
    candidates = torch.zeros((b, k, 2), dtype=torch.float32,device=device)  # 初始化为全零，形状为 (b, k, 2)
    candidate_scores = torch.zeros((b, k), dtype=torch.float32,device=device)  # 初始化为全零，形状为 (b, k)
    valid_mask = torch.ones((b, k), dtype=torch.bool,device=device)  # 1为有效，0为padding

    for i in range(b):  # 遍历每一个batch
        idx = torch.nonzero(vertex_mask[i])  # 选取大于阈值的点的坐标 (n, 2)
        if len(idx) == 0:
            continue

        y = idx[:, 0].float()
        x = idx[:, 1].float()
        
        # off = voff_map[i, :, idx[:, 0], idx[:, 1]]  # 对应偏移量 (2, n)
        # dx, dy = off[0], off[1]
        
        candidate_vertex = torch.stack((x,y), dim=1)*scale_size  # (n, 2) x,y坐标 从vmap尺寸转为原图尺寸
        num_candidates = candidate_vertex.shape[0]

        scores = vertex_scores[i][vertex_mask[i]]  # 对应分数 (n,)
        if num_candidates > k:
            # 按分数排序
            sorted_indices = torch.argsort(scores, descending=True)
            candidates[i] = candidate_vertex[sorted_indices[:k]]  # 只取前 k 个        
            candidate_scores[i] = scores[sorted_indices[:k]]  # 赋值对应的分数
        else:
            candidates[i, :num_candidates] = candidate_vertex  # 填充有效候选顶点
            candidate_scores[i, :num_candidates] = scores  # 赋值对应的分数
            valid_mask[i, num_candidates:] = 0  # 设置 padding点 为 0

    return dict(vertices=candidates, valid_mask=valid_mask,score=candidate_scores)#(b,k,2) (b,k)

def get_angle_between_vectors(v1, v2,arc=False):
    # compute angle in counter-clockwise direction between v1 and v2  逆时针转为正
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(det, dot)
    # Handle nan values
    if np.isnan(angle):
        # print("angel",0)
        angle = 0.0
    if arc:
        return angle#输出弧度
    else:
        return angle/np.pi*180 #输出角度
def generate_polygon1(mask_pred, junctions,inner_hole=True,max_distance=5):#prop为mask的一个连通域 改多边形策略
    #junctions为该图像中所有的顶点坐标列表(n,2)
    #inner_hole为True时，表示该连通域可能含内部孔洞
    im_h, im_w = mask_pred.shape
    contours, hierarchy = cv2.findContours(mask_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly = []
    for contour, h in zip(contours, hierarchy[0]):#contour为一个连通域的边界点坐标列表
        if inner_hole:
            if h[3] == -1:
                contour = ext_c_to_poly_coco(contour, im_h, im_w)
            if h[3] != -1:
                if cv2.contourArea(contour) >= 50:
                    contour = inn_c_to_poly_coco(contour, im_h, im_w)
                #c = inn_c_to_poly_coco(contour, im_h, im_w)
            if len(contour.shape)>2:
                contour=contour.squeeze(1)
        else:
            contour = np.array([c.reshape(-1).tolist() for c in contour])
        if len(contour) > 3:
            contour = simple_polygon(contour, thres=10)#简化mask轮廓的顶点
        init_poly = contour.copy()#init_poly为简化后的mask轮廓的顶点 若后面有junctions则用junctions中的顶点替换init_poly中的顶点
        if len(junctions) > 0:
            cj_match_ = np.argmin(cdist(contour, junctions), axis=1)#cdist计算两个集合中点的距离
            cj_dis = cdist(contour, junctions)[np.arange(len(cj_match_)), cj_match_]#cj_dis为contour中每个点到junctions中最近点的距离
            u, ind = np.unique(cj_match_[cj_dis < max_distance], return_index=True)#u为contour中距离小于5的点在junctions中的索引 ind为
            if len(u) > 2:
                ppoly = junctions[u[np.argsort(ind)]]#选出junctions中距离contour中最近的点
                ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                init_poly = ppoly
        init_poly = simple_polygon(init_poly, thres=10)
        poly.extend(init_poly.tolist())
    return np.array(poly)
def mask_guided_vertex_connect(mask, Vinit,max_distance=12):
    """
    mask: ndarray(h,w) 二值掩码
    Vinit: ndarray(n,2) 候选点坐标
    max_distance: int 最大距离
    return:
    polygon: ndarray(m,2) 筛选和排列后的顶点列表
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    im_h, im_w = mask.shape
    polygon = []
    low_iou_num=0#记录mask和预测的polygon的iou小于iou_thr的数量
    iou_thr=0.7
    for contour, h in zip(contours, hierarchy[0]):
        C = contour[:,0,:]
        if h[3] == -1:
            C = ext_c_to_poly_coco(contour, im_h, im_w)
        if h[3] != -1:
            if cv2.contourArea(contour) >= 50:
                C = inn_c_to_poly_coco(contour, im_h, im_w)
        if len(C) > 3:
            C = simple_polygon(C, thres=10)
            init_poly = C.copy()#若下面没有检测到多边形就用mask轮廓
            if len(Vinit) > 2:
                dist=cdist(C, Vinit)#cdist计算两个集合中点的距离 
                Vmatch = np.argmin(dist, axis=1)
                #Vmatch[i]为距离C[i]最近的Vinit中点的下标 eg:[0,2,4,4,4,1,1...]
                cv_dis = dist[np.arange(len(Vmatch)), Vmatch]
                #cv_dis[i]为Vinit中距离C[i]最近的点的距离 eg:[距离(C[0],Vinit[0])，距离(C[1],Vinit[2]...]
                # 删除Vmatch中连续的重复，并保存重复点中距离C中点的最小值：
                min_cv_dis=[]#记录重复点中距离C中点的最小值
                Vmatch_u=[]#记录去除Vmatch中连续重复后Vinit中的点下标
                Cclosest=[]#记录匹配的C中的点 对应距离最小值位置
                for i, k in enumerate(Vmatch):
                    if i == 0 or k != Vmatch[i - 1]:#第一个点或者不等于前一个点
                        Vmatch_u.append(k)
                        Cclosest.append(i)
                        min_cv_dis.append(cv_dis[i])
                    else:
                        if cv_dis[i]<min_cv_dis[-1]:#若距离小于最小值
                            min_cv_dis[-1]=cv_dis[i]
                            Cclosest[-1]=i
                i=0
                while i<len(Cclosest):
                    last_i=(i-1)%len(Cclosest)
                    start = Cclosest[last_i]
                    end = Cclosest[i]
                    vec_contour=C[end]-C[start]
                    start_v = Vmatch_u[last_i]
                    end_v = Vmatch_u[i]
                    vec_vertex=Vinit[end_v]-Vinit[start_v]
                    angle=get_angle_between_vectors(vec_contour,vec_vertex)#计算两向量夹角
                    if np.abs(angle/10)+min_cv_dis[i]>max_distance:
                        Cclosest.pop(i)
                        Vmatch_u.pop(i)
                        min_cv_dis.pop(i)
                        i-=1
                    i+=1
                if len(Vmatch_u)>2:
                    init_poly=Vinit[Vmatch_u]
                    init_poly = simple_polygon(init_poly, thres=10)
                    if len(init_poly) < 3:
                        polygon.extend(C.tolist())
                        low_iou_num=1
                        continue                    
                    if iou_poly(init_poly,C)<iou_thr:
                        low_iou_num=1
            
            polygon.extend(init_poly.tolist())
    return np.array(polygon),low_iou_num

def ext_c_to_poly_coco(ext_c, im_h, im_w):
    mask = np.zeros([im_h+1, im_w+1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_mask[f_y + 1, f_x] = 1
    trans_mask[f_y, f_x + 1] = 1
    trans_mask[f_y + 1, f_x + 1] = 1
    contours, _ = cv2.findContours(trans_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly

def diagonal_to_square(poly):
    new_c = []
    for id, p in enumerate(poly[:-1]):
        if (p[0] + 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]) \
                or (p[0] - 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
        elif (p[0] + 1 == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])
    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly

def inn_c_to_poly_coco(inn_c, im_h, im_w):
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    trans_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0
    #trans_mask[max(f_y), max(f_x)] = 1
    contours, _ = cv2.findContours(trans_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    #return poly
    new_poly = diagonal_to_square(poly)
    return new_poly

def simple_polygon(poly, thres=10,near_spine_thres=(175, 185)):#poly:多边形的顶点坐标ndarray(n,2)
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]#如果多边形的第一个顶点和最后一个顶点相同，则将最后一个顶点删除
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]#计算多边形每条边的向量
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    # 将多边形顶点坐标中两边夹角（顺着轮廓遍历的方向）小于thres度，和当两边夹角大于175,小于185（近乎尖刺）的筛掉
    flag_low_angle = (lines_ang > thres) & (lines_ang < near_spine_thres[0])
    flag_high_angle = (lines_ang > near_spine_thres[1]) & (lines_ang < 360 - thres)
    flag = flag_low_angle | flag_high_angle

    # Select vertices that are not near straight angles
    simple_poly = poly[np.roll(flag, 1, axis=0)]
    if len(simple_poly) < 3:
        return simple_poly
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1,2)))#尾部加上头部顶点
    return simple_poly