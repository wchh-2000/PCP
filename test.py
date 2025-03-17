import os,json
join = os.path.join
# torch.manual_seed(42)
# np.random.seed(42)
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
#self defined:
from model import PromptModel
from dataset.dataset_crop import PromptDataset,collate_fn_test
from utils.arg_utils import load_args
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='large')
parser.add_argument('--res_dir', type=str, default='res_prompt')
parser.add_argument('--work_dir', type=str, default='work_dir_loveda')
parser.add_argument('--gpu',type=int, default=0)
#model config:
parser.add_argument('--model_type', type=str, default='vit_b',help='for image encoder')
parser.add_argument('--freeze_img', type=bool, default=False,help='whether to freeze image encoder weights')
parser.add_argument('--freeze_mask', type=bool, default=False,help='whether to freeze mask decoder weights')
parser.add_argument('--upconv',type=bool,default=True,help='whether to use upsample and conv to upsample vmap voff')
parser.add_argument('--multi_mask',type=bool,default=True,help='whether to predict multi-mask')
parser.add_argument('--train_cls',type=bool,default=False, help='whether train vertex classify model')
parser.add_argument('--dist_thr',type=int,default=15, help='distance threshold for pred and gt vertex matching')

parser.add_argument('--image_size', type=int, default=224,help='input image size to the model, a multiple of 16')
parser.add_argument('--eval',type=bool,default=False, help='whether to evaluate the metrics')
parser.add_argument('--save_metric_data',type=bool,default=False)
#data config:
parser.add_argument('--dataset', type=str, default='loveda')
parser.add_argument('--large', type=bool,default=False)
parser.add_argument('--ann_file', type=str,default='ann_large_prompt.json')
parser.add_argument('--img_dir', type=str,default='images')
parser.add_argument('--add_edge',type=bool,default=True,help='whether to add edge(boundary) prediction')
parser.add_argument('--gaussian', type=bool, default=True,help='whether to use gaussian kernel to generate vertex confidence map')
#prompt config:
parser.add_argument('--bbox', type=bool, default=True,help='whether to use bbox as prompt')
parser.add_argument('--mask_prompt', type=bool, default=True,help='whether to use mask as prompt')
parser.add_argument('--crop_noise', type=bool, default=False,help='whether to add noise to the crop area')

#Load the same arguments from the training arguments (ignore the 'gpus' argument):
args = load_args(parser)
args.result_pth=f'{args.work_dir}/{args.task_name}/{args.res_dir}/'
args.checkpoint=f'{args.work_dir}/{args.task_name}/version_0/checkpoints/bestIoU.ckpt' #loveda_small.pth loveda_large.pth
args_dict = vars(args)
print(args_dict)
os.makedirs(args.result_pth, exist_ok=True)
#set Dataset:
dataset_param=dict(input_size=args.image_size,
                   bbox=args.bbox,mask_prompt=args.mask_prompt,#prompt type
                   large=args.large,
                   add_edge=args.add_edge,gaussian=args.gaussian)
dataset_pth = dict(data_root=f'dataset/{args.dataset}/test', ann_file=args.ann_file, img_dir=args.img_dir)
#loveda:
if args.dataset=='loveda':
    dataset_pth = dict(data_root=f'dataset/{args.dataset}/Val/Rural/', ann_file=args.ann_file, img_dir='images_png')

dataset_param['crop_noise']=args.crop_noise
batch_size=60
num_workers=6
dataset = PromptDataset(dataset_pth, mode='test',load_gt=args.eval,**dataset_param)
dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False,
                    collate_fn=collate_fn_test,pin_memory=True)

class TestConfig:
    def __init__(self):
        self.train=False
        self.eval=args.eval
        self.log=False
        self.save_results=True
test_cfg=TestConfig()
device = 'cuda:'+str(args.gpu)

model=PromptModel.load_from_checkpoint(args.checkpoint,args=args,test_cfg=test_cfg,map_location=device)
model.eval()
low_iou_num=0
for step, batch in enumerate(tqdm(dataloader)):
    # if step>10:
    #     break
    batch=model.transfer_batch_to_device(batch,device,step)
    low_iou_num+=model.validation_step(batch, step,log=False)
N=len(dataset)
ratio=round(low_iou_num/N*100,3)
print(f"low iou ratio:{ratio}%")
with open(join(args.result_pth,'low_iou_ratio.txt'),'w') as f:
    f.write(f"polygon, mask low iou ratio:{ratio}%")
# print("avg process time:",round(model.avg_process_time/N*1000,2),"ms",
#       "avg pos process time:",round(model.avg_pos_process_time/N*1000,3),"ms")
if args.eval:
    if args.save_metric_data:
        import pickle
        data={'m':model.metrics_calculator,'N':N}
        with open(join(args.result_pth, f'metrics_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
    metrics = model.metrics_calculator.compute_average(N)
    for key in metrics:
        metrics[key] = round(metrics[key] * 100, 2)
    print(metrics)
    csv_file_path = join(args.result_pth, f'metrics.csv')
    file_exists = os.path.isfile(csv_file_path)        
    with open(csv_file_path, 'a+', newline='') as csvfile:
        fieldnames = ['exp']+list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 如果文件不存在，则写入表头
        # if not file_exists:
        writer.writeheader()
        # task = args.task_name.split('/')[1]
        content={'exp': ''}
        content.update(metrics)
        writer.writerow(content)
    
if test_cfg.save_results:
    name=f'results_polyon.json'
    dt_file=join(args.result_pth,name)    
    with open(dt_file,'w') as _out:
        json.dump(model.results_poly,_out)
    print("Polygon results save to:",dt_file)#coco format