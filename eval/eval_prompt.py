from eval.iou import iou_from_poly
from eval.vertex_set_pr import precision_recall_from_vertex_set
class PromptMetrics:
    def __init__(self):
        self.metrics = {'miou': 0, 'vf1': 0,'v_precision': 0, 'v_recall': 0}

    def calculate_metrics(self, pred_polygon, gt_polygon):
        p, r = precision_recall_from_vertex_set(pred_polygon[:-1, :], gt_polygon[:-1, :])#最后一个点为重复点，去除
        vf1 = 2 * p * r / (p + r + 1e-8)
        miou = iou_from_poly(pred_polygon, gt_polygon, 512, 512)#first to mask.
        self.update_metrics(self.metrics, p, r, miou, vf1)

    def update_metrics(self, metrics, p, r, miou, f1):
        metrics['v_precision'] += p
        metrics['v_recall'] += r
        metrics['miou'] += miou
        metrics['vf1'] += f1

    def average_metrics(self, metrics, count):
        if count > 0:
            for key in metrics:
                metrics[key] /= count

    def compute_average(self,n=None):        
        self.average_metrics(self.metrics, n)#no mask分数记为0，总数不排除no mask
        m=self.metrics.copy()
        self.metrics = {'miou': 0, 'vf1': 0,'v_precision': 0, 'v_recall': 0}
        return m
