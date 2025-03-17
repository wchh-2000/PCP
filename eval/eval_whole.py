"""
评价coco格式的结果json的各类cIoU, 实例vertex precision, recall, f1指标
使用base环境
"""
import os,csv
from os.path import join
import sys
sys.path.append(os.path.abspath('./'))#add project root to PATH
from cIoU import compute_IoU_cIoU
from shape_acc import VertexPR
def save_csv(out_file,fields,content):
    file_exists = os.path.isfile(out_file)
    with open(out_file,'a') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(content)
    print(f"Metrics saved to {out_file}")
def cal_metrics(pred,gt,exp_name,outdir,Tv=6,
                value_mapping={1: 'build'},                
                category_ids=[1],
                ignore_index=None,
                metric=set(['cIoU','vertex'])):
    if 'cIoU' in metric:
        m=compute_IoU_cIoU(pred,gt,
                        value_mapping,category_ids,ignore_index)
        out_file=join(outdir,f'cIoU.csv')
        fields = ['exp']
        content={'exp':exp_name}         
        for class_id in value_mapping.keys():
            name=value_mapping[class_id]
            for k,v in m[class_id].items():
                fields.append(f'{name}_{k}')
                content[f'{name}_{k}']=v
        save_csv(out_file,fields,content)
    if 'vertex' in metric:
        out_file_v=join(outdir,'vertex.csv')
        fields_v=['exp']
        content_v={'exp':exp_name}
        metrics_calculator = VertexPR(pred, gt,Tv)
        m=metrics_calculator.calculate_metrics(category_ids)

        for class_id in value_mapping.keys():
            name=value_mapping[class_id]
            name_p=f'{name}_p'
            name_r=f'{name}_r'
            fields_v.extend([name_p,name_r])
            vp=m[class_id]['vertex_precision']
            vr=m[class_id]['vertex_recall']
            content_v[name_p]=vp
            content_v[name_r]=vr
            # if name=='overall':
            name_f1=f'{name}_f1'
            fields_v.append(name_f1)
            vf1=(vp*vr*2)/(vp+vr+1e-8)
            content_v[name_f1]=round(vf1,2)
        save_csv(out_file_v,fields_v,content_v)

def eval_loveda():
    args=parse_args()
    value_mapping={1: 'build'}
    category_ids=[1]
    ignore_index=None#0
    Tv=6
    for res,exp in zip(args.res,args.exps):
        cal_metrics(res,args.gt,exp,args.outdir,Tv,value_mapping,
            category_ids,ignore_index,metric=set(['cIoU','vertex']))
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res',default=[
                                'dataset/loveda/Val/Rural/res/seg.json',
                                'work_dir_loveda/merge_res.json'])
    parser.add_argument('--exps',default=['SFA-Net+DP','SFA-Net+PCP'])
    parser.add_argument('--gt',default='dataset/loveda/Val/Rural/ann_build.json')
    parser.add_argument('--outdir',default='work_dir_loveda')
    args = parser.parse_args()
    return args
if __name__ == "__main__":    
    eval_loveda()
