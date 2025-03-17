from shapely.geometry import Polygon,MultiPolygon
from shapely.strtree import STRtree
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
from dataset.data_utils import cocoSeg2polygon
from vertex_set_pr import precision_from_vertex_set,recall_from_vertex_set
def coco_to_polygons(file,gt=False,simp=None):
    """
    将COCO格式的annotation转为Shapely的多边形。
    """
    with open(file, "r") as f:
        coco_anns = json.load(f)
    polygons_by_class = defaultdict(lambda: defaultdict(list))
    # {category_id: {image_id: [Polygon, ...], ...}, ...}
    if gt:
        coco_anns=coco_anns['annotations']
    for ann in coco_anns:
        category_id = ann["category_id"]
        image_id = ann["image_id"]
        poly=cocoSeg2polygon(ann["segmentation"])
        if simp is not None:
            poly=poly.simplify(simp)
        polygons_by_class[category_id][image_id].append(poly)
    return polygons_by_class

def polygon2ndarray(polygon):  
    """  
    将shapely的Polygon或MultiPolygon对象转换为ndarray。  
    """  
    if isinstance(polygon, Polygon):
        exterior_coords = list(polygon.exterior.coords)  
        interior_coords = [list(interior.coords) for interior in polygon.interiors]  
        vertices = np.array(exterior_coords)  
        for interior in interior_coords:  
            vertices = np.concatenate((vertices, np.array(interior)), axis=0)  
    elif isinstance(polygon, MultiPolygon):
        vertices = []  
        for poly in polygon.geoms:  
            exterior_coords = list(poly.exterior.coords)  
            interior_coords = [list(interior.coords) for interior in poly.interiors]  
            poly_vertices = np.array(exterior_coords)  
            for interior in interior_coords:  
                poly_vertices = np.concatenate((poly_vertices, np.array(interior)), axis=0)  
            vertices.append(poly_vertices)
        vertices = np.vstack(vertices) if vertices else np.array([])  
    else:  
        raise TypeError("Input must be a Polygon or MultiPolygon object")  
      
    return vertices

class VertexPR:
    # vertex precision and recall
    def __init__(self, pred, gt,Tv=3,simp=None):
        """
        pred和gt格式应为COCO格式的annotations。
        """
        self.pred_by_class = coco_to_polygons(pred)
        self.gt_by_class = coco_to_polygons(gt,gt=True,simp=simp)
        self.Tv=Tv
        self.simple_thr={1:90}
        #简单多边形的点数阈值,只计算简单多边形的点集精度和召回
        self.total_vp=0
        self.total_vr=0
        self.total_pred_simp=0
        self.total_gt_simp=0

    def calculate_precision(self, class_id):
        """
        计算精确率：检测到的pred图斑和gt重叠部分的比例超过阈值则算正确。
        """
        preds_by_image = self.pred_by_class.get(class_id, {})
        gts_by_image = self.gt_by_class.get(class_id, {})

        vertex_precision = 0
        simple_polygon_num = 0

        for image_id, preds in preds_by_image.items():
            # if image_id>20:
            #     break
            gts = gts_by_image.get(image_id, [])
            
            # 建立空间索引
            tree = STRtree(gts)

            for pred in preds:
                pred_np=polygon2ndarray(pred)
                simple=True if len(pred_np)<self.simple_thr[class_id] else False
                overlapped_area=0
                gt_overlapped=[]
                for gt_id in tree.query(pred):
                    gt = gts[gt_id]
                    area=pred.intersection(gt).area
                    overlapped_area +=area
                    if simple and area>0:
                        gt_overlapped.append(polygon2ndarray(gt))#与该pred有重叠的gt构成的点集
                #点集的精度：
                if simple:
                    if len(gt_overlapped)>0:
                        gt_overlapped=np.concatenate(gt_overlapped,axis=0)
                        vertex_precision+=precision_from_vertex_set(pred_np,gt_overlapped,threshold=self.Tv)
                        #未匹配到gt时该pred忽略
                        simple_polygon_num+=1
        vp_m=vertex_precision/simple_polygon_num if simple_polygon_num!=0 else 1.0
        self.total_vp+=vertex_precision
        self.total_pred_simp+=simple_polygon_num

        return vp_m

    def calculate_recall(self, class_id):
        """
        计算召回率：每个gt的重叠比例超出阈值即为正确召回。
        """
        preds_by_image = self.pred_by_class.get(class_id, {})
        gts_by_image = self.gt_by_class.get(class_id, {})

        vertex_recall=0
        simple_polygon_num=0

        for image_id, gts in gts_by_image.items():
            # if image_id>20:
            #     break
            preds = preds_by_image.get(image_id, [])        
            # 建立空间索引
            tree = STRtree(preds)
            for gt in gts:
                gt_np=polygon2ndarray(gt)
                simple=True if len(gt_np)<self.simple_thr[class_id] else False
                overlapped_area=0
                pred_overlapped=[]
                for pred_id in tree.query(gt):
                    pred = preds[pred_id]
                    area=gt.intersection(pred).area
                    overlapped_area +=area
                    if simple and area>0:
                        pred_overlapped.append(polygon2ndarray(pred))
                #点集的召回：
                if simple:
                    if len(pred_overlapped)>0:
                        pred_overlapped=np.concatenate(pred_overlapped,axis=0)
                        vertex_recall+=recall_from_vertex_set(pred_overlapped,gt_np,threshold=self.Tv)
                        #未匹配到pred时该gt忽略
                        simple_polygon_num+=1
        vr_m=vertex_recall/simple_polygon_num if simple_polygon_num!=0 else 1.0
        self.total_vr+=vertex_recall
        self.total_gt_simp+=simple_polygon_num
        return vr_m

    def calculate_metrics(self,all_class_ids):
        """
        计算所有类别的精确率和召回率。
        """
        class_metrics = {}
        # all_class_ids = set(self.pred_by_class.keys()).union(set(self.gt_by_class.keys()))
        for class_id in tqdm(all_class_ids):
            vp = self.calculate_precision(class_id)
            vr = self.calculate_recall(class_id)
            vp=round(vp*100,2)
            vr=round(vr*100,2)
            class_metrics[class_id] = {'vertex_precision':vp,'vertex_recall':vr}
        vp_avg=self.total_vp/self.total_pred_simp
        vr_avg=self.total_vr/self.total_gt_simp
        vp_avg=round(vp_avg*100,2)
        vr_avg=round(vr_avg*100,2)
        class_metrics['overall']={'vertex_precision':vp_avg,'vertex_recall':vr_avg}
        return class_metrics


