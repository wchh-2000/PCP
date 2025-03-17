import torch
import torch.nn as nn
import cv2
from scipy.spatial.distance import cdist
import numpy as np
from utils.polygon import simple_polygon
from eval.iou import iou_poly
class SRVFC(nn.Module):
    # Shape-Rule-Based Vertex Filtering and Connecting Module
    def __init__(self,large=False):
        super(SRVFC, self).__init__()
        self.large=large
        self.dis_ratio = 3#10 if large else 3
        self.sample_interval = 8 if large else 4
        self.Tdist = 20 if large else 15
        n_features=2
        self.output_proj = nn.Linear(n_features, 1)
        
        weights = self.output_proj.weight.data
        # 初始化权重为 [0, 1]（按行初始化）
        weights[0, 0] = -0.5  # 设置第一个元素为 -0.5 距离特征
        weights[0, 1] = 0.5  # 设置第二个元素为 0.5 形状特征 正比
        self.output_proj.bias.data.fill_(0)

    def vertex_connect(self, vertices, seg_logit,predict=False):
        """
        vertices: dict vertices:[b, n, 2] n个点的坐标 W*W图像坐标 原图尺寸
                   valid_mask:[b, n] 1为有效，0为padding    
        seg_logit:[b,W,W] 分割logit图
        return shape_features:[b,n,2] n个点的两个特征，分别为距离contour最近的点的距离，以及最近点的方向变化率（0~pi）
        """
        vertices,valid_mask=vertices['vertices'],vertices['valid_mask']
        if predict:
            polygon=[]
            mask_contours=[]
            low_iou_num=0#记录mask和预测的polygon的iou小于iou_thr的数量
            iou_thr=0.7
            valid_polygon=np.zeros(vertices.shape[0], dtype=bool)  # 初始化有效性标记为False
        else:
            shape_features=torch.zeros(vertices.shape[0],vertices.shape[1],2,device=vertices.device)
        vertices=vertices.detach().cpu().numpy()
        valid_mask=valid_mask.cpu().numpy()
        seg_logit=seg_logit.detach().cpu().numpy()
        for b in range(vertices.shape[0]):
            valid_index=valid_mask[b] > 0
            Vinit = vertices[b][valid_index]
            if len(Vinit)==0:
                if predict:
                    polygon.append(None)
                else:
                    shape_features[b,valid_index]=0
                continue
            seg_mask = (seg_logit[b] > 0).squeeze().astype(np.uint8)
            # 查找外轮廓，暂不考虑内部孔洞
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#cv2.CHAIN_APPROX_SIMPLE

            if len(contours)>1:
                # 通过面积找出最大的轮廓
                contour = max(contours, key=cv2.contourArea)
            elif len(contours)!=0:
                contour=contours[0]
            else:
                if predict:
                    polygon.append(None)
                else:
                    shape_features[b,valid_index]=0
                continue
            contour = contour.squeeze()
            if len(contour)<3:
                if predict:
                    polygon.append(None)
                else:
                    shape_features[b,valid_index]=0
                continue
            contour=contour[0::self.sample_interval,:]#均匀采样 todo small试验
            # contour = np.array(Polygon(contour).simplify(1).exterior.coords)#简化轮廓 large
            # contour=contour[:-1,:]#去掉最后一个点，与第一个点重复
            dist=cdist(contour, Vinit)#cdist计算两个集合中点的距离
            if not predict:#get shape features
                Cmatch = np.argmin(dist, axis=0)
                #Cmatch[i]为距离Vinit[i]最近的contour中点的下标 长度与Vnit一致 eg:[0,2,4,4,4,1,1...]
                direct_change=direction_change_rate(contour, Cmatch)
                vc_dis = dist[Cmatch, np.arange(len(Vinit))]/self.dis_ratio #调整幅度
                shape_features[b,valid_index,0]=torch.tensor(vc_dis,device=shape_features.device,dtype=torch.float32)
                shape_features[b,valid_index,1]=torch.tensor(direct_change,device=shape_features.device,dtype=torch.float32)
            else:#get polygons                
                mask_contours.append(contour)
                Vmatch = np.argmin(dist, axis=1)#Vmatch[i]为距离contour[i]最近的Vinit中点的下标 eg:[0,2,4,4,4,1,1...]
                Vmatch_u=[]#记录去除Vmatch中连续重复后Vinit中的点下标
                cv_dis = dist[np.arange(len(Vmatch)), Vmatch]
                #cv_dis[i]为Vinit中距离C[i]最近的点的距离 eg:[距离(C[0],Vinit[0])，距离(C[1],Vinit[2]...]
                min_cv_dis=[]#记录重复点中距离C中点的最小值
                for i, k in enumerate(Vmatch):
                    if i == 0 or k != Vmatch[i - 1]:#第一个点或者不等于前一个点
                        Vmatch_u.append(k)
                        min_cv_dis.append(cv_dis[i])
                    else:
                        if cv_dis[i]<min_cv_dis[-1]:#若距离小于最小值
                            min_cv_dis[-1]=cv_dis[i]
                Vmatch_u=remove_duplicates(Vmatch_u, min_cv_dis)
                if len(Vmatch_u)>2:
                    init_poly=Vinit[Vmatch_u]
                    init_poly = simple_polygon(init_poly, thres=10)
                    if len(init_poly)>2:
                        polygon.append(init_poly)
                        valid_polygon[b]=True
                        if iou_poly(init_poly,contour)<iou_thr:
                            low_iou_num+=1
                    else:
                        polygon.append(None)
                else:
                    polygon.append(None)
        if predict:
            return polygon,valid_polygon,low_iou_num,mask_contours
        else:
            return shape_features

    def forward(self, vertices,seg_logit):
        """
        vertices: dict vertices:[b, n, 2] n个点的坐标 W*W图像坐标 原图尺寸
                   valid_mask:[b, n] 1为有效，0为padding    
        seg_logit:[b,h,w] 分割logit图
        return: [b, n, 1] 点的分类概率
        """
        shape_feature = self.vertex_connect(vertices, seg_logit)        
        x = self.output_proj(shape_feature)
        return x.squeeze(-1)
    def predict(self, vertices,seg_logit):
        """
        vertices: dict vertices:[b, n, 2] n个点的坐标 W*W图像坐标 原图尺寸
                   valid_mask:[b, n] 1为有效，0为padding    
        seg_logit:[b,h,w] 分割logit图
        return: polygons list ndarray m,2 m不定 W*W图像坐标
        """
        vertex_conf=self.forward(vertices,seg_logit)# [b, n]
        vertex_prob=torch.sigmoid(vertex_conf)
        vertex_pred=torch.zeros_like(vertices['vertices'])
        vertex_pred_mask=torch.zeros_like(vertex_conf)
        scores=[]
        for b in range(vertex_conf.shape[0]):
            pos=vertex_conf[b]>0 #before sigmoid
            n=sum(pos)
            vertex_pred[b,:n]=vertices['vertices'][b,pos,:]
            vertex_pred_mask[b,:n]=1
            scores.append(vertex_prob[b,pos].mean().item())
        vertex_pred_dict={'vertices':vertex_pred,'valid_mask':vertex_pred_mask}
        polygon_result=self.vertex_connect(vertex_pred_dict, seg_logit,predict=True)
        return polygon_result,np.array(scores)
    def get_target(self,vertex_pred,vertex_gt):
        """
        vertex_pred: dict vertices:[b, n, 2] n个点的坐标 W*W图像坐标 原图尺寸
                   valid_mask:[b, n] 1为有效，0为padding    
        vertex_gt:list ndarray m,2 m不定 W*W图像坐标
        return: [b, n, 1] 点的分类标签
        """
        vertex_pred,valid_mask=vertex_pred['vertices'],vertex_pred['valid_mask']
        device=vertex_pred.device
        vertex_pred=vertex_pred.detach()
        valid_mask=valid_mask
        B, n, _ = vertex_pred.shape
        target=torch.zeros(B, n,device=device)
        target_mask = torch.zeros(B, n,device=device)#用于计算loss的权重矩阵 1为有效，0为padding
        for b in range(B):
            valid_index=valid_mask[b] > 0
            vertex_gt_b=torch.tensor(vertex_gt[b],device=device,dtype=torch.float32)
            vertex_pred_b=vertex_pred[b][valid_index]
            # 如果没有有效点则跳过
            if vertex_pred_b.shape[0] == 0 or len(vertex_gt_b) == 0:
                continue
            # 计算距离矩阵
            distances = torch.cdist(vertex_pred_b, vertex_gt_b)
            # 找到每个真值点最近的预测点索引
            nearest_pred = torch.argmin(distances, dim=0)
            # 将对应的预测点匹配，分类标签置为1
            # target[b, nearest_pred] = 1
            for gt_i,pred_i in enumerate(nearest_pred):
                if distances[pred_i,gt_i] < self.Tdist:
                    # 将对应的预测点匹配，分类标签置为1
                    target[b, pred_i] = 1

            # 更新有效mask，未匹配的点依然保持为0
            target_mask[b, valid_index] = 1

        return target,target_mask
def dis_gt_conf(dis,dis1=15,dis0=20):
    #三段函数
    if dis<dis1:
        return 1
    elif dis<dis0:
        return -(dis-dis1)/(dis0-dis1)+1
    else:
        return 0

    
def remove_duplicates(Vmatch_u, min_cv_dis):
    #Vmatch_u中出现重复时，取min_cv_dis最小的，去除其余重复
    n = len(Vmatch_u)
    if n <= 2:
        return Vmatch_u, min_cv_dis
    # Map each value in Vmatch_u to its indices and corresponding distances
    value_to_indices = {}
    for idx in range(n):
        val = Vmatch_u[idx]
        dis = min_cv_dis[idx]
        if val not in value_to_indices:
            value_to_indices[val] = [(idx, dis)]
        else:
            value_to_indices[val].append((idx, dis))

    # Determine indices to keep based on the smallest distance
    indices_to_keep = set()
    for val, idx_dis_list in value_to_indices.items():
        # Keep the index with the smallest distance
        min_idx = min(idx_dis_list, key=lambda x: x[1])[0]
        indices_to_keep.add(min_idx)

    # Build the new lists by including only the selected indices
    new_Vmatch_u = [Vmatch_u[idx] for idx in sorted(indices_to_keep)]
    # new_min_cv_dis = [min_cv_dis[idx] for idx in sorted(indices_to_keep)]

    return new_Vmatch_u#, new_min_cv_dis
def direction_change_rate(contour, m):
    """
    计算轮廓上点m处的方向变化率，取m前后各k个点处的方向向量，计算两个向量的夹角。
    取k=1,2的平均夹角作为m点的方向变化率。
    
    参数：
    contour: np.ndarray, shape (n, 2) - 轮廓的坐标点
    m: ndarray,下标列表 要计算的目标点的索引
    
    返回：
    float - 点 m 处的方向变化率（以弧度表示）
    """
    n=len(contour)    
    # 获取 m 点前后的两个点
    def cal_diff(k):
        pf = contour[(m - k)%n]
        p = contour[m]
        pb = contour[(m + k)%n]
        
        # 计算两条线段的方向向量
        vec1 = p - pf  # m 前的线段方向向量
        vec2 = pb - p  # m 后的线段方向向量
        
        # 计算每个向量的方向角（单位为弧度）
        # arctan2(y, x) 返回的是 y/x 的反正切值，值域[-π, π]
        angle1 = np.arctan2(vec1[:,1], vec1[:,0])  # 前段的方向角
        angle2 = np.arctan2(vec2[:,1], vec2[:,0])  # 后段的方向角
        
        delta_angle = angle2 - angle1
        # 确保弧度差在 [-π, π] 范围内 -> abs [0, π]:
        delta_angle = abs(np.arctan2(np.sin(delta_angle), np.cos(delta_angle)))
        return delta_angle
    delta_angle=sum([cal_diff(k) for k in [1,3]])/2#1,3 todo change
    
    return delta_angle
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
        return angle#输出弧度[-π, π]
    else:
        return angle/np.pi*180 #输出角度[-180°, 180°]