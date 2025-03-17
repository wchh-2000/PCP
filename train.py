import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
# import numpy as np
# torch.manual_seed(42)
# np.random.seed(42)
from torch.utils.data import DataLoader
import argparse,json,os
def str2dict(v):
    try:
        return json.loads(v)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid JSON format for --loss_weight")
from model import PromptModel
from dataset.dataset_crop import PromptDataset,collate_fn_test
from utils.arg_utils import load_train_args
join = os.path.join
debug=False
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='small')
#training config:
parser.add_argument('--config', type=str, default='configs/loveda_small.json',
                    help='in func load_train_args, use config file to update args')
parser.add_argument('--work_dir', type=str, default='work_dir_loveda')
parser.add_argument('--gpus',type=int, nargs='+', default=[0])#[0,1]
parser.add_argument('--distributed', action='store_true',help='default False')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--batch_size_val', type=int, default=60)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--img_encoder_lr', type=float, default=5e-6)
parser.add_argument('--decoder_lr', type=float, default=5e-5)
parser.add_argument('--cls_lr', type=float, default=1e-3)
parser.add_argument('--iter_dec_lr', type=float, default=1e-3)
parser.add_argument('--lr_milestone', type=float, default=12,help='start epoch to decay lr by cosine annealing')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--val_epoch', type=int, default=4,help='validation frequency, every n epochs')
parser.add_argument('--loss_weight', type=dict, default=dict(vmap=1.25,voff=5,mask=1,edge=0.5,iou=5,cls=2))
                    # type=str2dict, required=True)
#model config:
parser.add_argument('--model_type', type=str, default='vit_b',help='for image encoder')
parser.add_argument('--checkpoint', type=str, default='segment_anything/sam_vit_b_01ec64.pth')
parser.add_argument('--freeze_img', type=bool, default=False,help='whether to freeze image encoder weights')
parser.add_argument('--freeze_mask', type=bool, default=False,help='whether to freeze mask decoder weights')
parser.add_argument('--upconv',type=bool,default=True,help='whether to use upsample and conv to upsample vmap voff')
parser.add_argument('--multi_mask',type=bool,default=True,help='whether to predict multi-mask')
parser.add_argument('--use_pretrained',type=bool,default=False,help='whether to use pretrained small model when training large model')
parser.add_argument('--train_cls',type=bool,default=True, help='whether train vertex classify model(SRVFC)')
parser.add_argument('--train_cls_start',type=int,default=8, help='start training cls model on epoch train_cls_start')
parser.add_argument('--dist_thr',type=int,default=15, help='distance threshold for pred and gt vertex matching')

parser.add_argument('--image_size', type=int, default=224,help='input image size to the model, a multiple of 16')
#data config:
parser.add_argument('--dataset', type=str, default='loveda')
parser.add_argument('--large', type=bool,default=False)
parser.add_argument('--ann_file', type=str,default='ann.json')
parser.add_argument('--add_edge',type=bool,default=True,help='whether to add edge(boundary) prediction')
parser.add_argument('--gaussian', type=bool, default=True,help='whether to use gaussian kernel to generate vertex confidence map')
#prompt config:
parser.add_argument('--bbox', type=bool, default=True,help='whether to use bbox as prompt')
parser.add_argument('--mask_prompt', type=bool, default=True,help='whether to use mask as prompt')
parser.add_argument('--crop_noise', type=bool, default=False)
args = load_train_args(parser)
if debug:
    args.task_name='debug'
    # args.gpus=[0]
    args.distributed=False
    args.val_epoch=1
if args.distributed:
    args.img_encoder_lr*=len(args.gpus)
    args.decoder_lr*=len(args.gpus)

args.log_dir = join(args.work_dir, args.task_name)
os.makedirs(args.log_dir, exist_ok=True)
args_dict = vars(args)
print(args_dict)
with open(join(args.log_dir,'args.json'), 'w') as f:
    json.dump(args_dict, f)

dataset_param=dict(input_size=args.image_size,
                        add_edge=args.add_edge,gaussian=args.gaussian,large=args.large,
                        bbox=args.bbox,mask_prompt=args.mask_prompt)
dataset_param['crop_noise']=args.crop_noise
train_dataset_pth = dict(data_root=f'dataset/{args.dataset}/train', ann_file=args.ann_file, img_dir='images')
val_dataset_pth = dict(data_root=f'dataset/{args.dataset}/test', ann_file=args.ann_file, img_dir='images')
if args.dataset=='loveda':
    train_dataset_pth = dict(data_root=f'dataset/{args.dataset}/Train/Rural/', ann_file=args.ann_file, img_dir='images_png')
    val_dataset_pth = dict(data_root=f'dataset/{args.dataset}/Val/Rural/', ann_file=args.ann_file, img_dir='images_png')
train_dataset = PromptDataset(train_dataset_pth,**dataset_param)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers,
                                collate_fn=collate_fn_test,shuffle=True,pin_memory=True)
val_dataset = PromptDataset(val_dataset_pth, mode='val',**dataset_param)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val,num_workers=args.num_workers,
                            collate_fn=collate_fn_test, shuffle=False,pin_memory=True)

class TestConfig:
    def __init__(self):
        self.train=True
        self.eval=True
        self.save_results=False
        self.log=True
test_cfg=TestConfig()
if 'sam_vit' not in args.checkpoint:
    device = 'cuda:'+str(args.gpus[0])
    model=PromptModel.load_from_checkpoint(args.checkpoint,args=args,test_cfg=test_cfg,strict=False,map_location=device)
    print(f'load from {args.checkpoint}')
else:
    model = PromptModel(args, test_cfg=test_cfg)
if args.distributed:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        every_n_epochs=args.val_epoch,
        save_last=True,
        save_top_k=1,
        filename='bestIoU'
        )
logger = TensorBoardLogger(args.work_dir, name=args.task_name)
train_param=dict(
    max_epochs=args.epochs,
    log_every_n_steps=50,
    devices=args.gpus,
    check_val_every_n_epoch=args.val_epoch,
    num_sanity_val_steps=0,
    # accelerator='cpu',
    logger=logger,
    default_root_dir=args.log_dir,
    callbacks=[checkpoint_callback])
if args.distributed:
    train_param.update(dict(
        accelerator="gpu", strategy="ddp_find_unused_parameters_true"))
if debug:
    train_param.update(dict(
        limit_train_batches=10,
        limit_val_batches=4,
        # check_val_every_n_epoch=1
        ))
trainer = pl.Trainer(**train_param)
trainer.fit(model, train_dataloader, val_dataloader)
