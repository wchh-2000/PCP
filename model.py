from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,ConstantLR, SequentialLR
import os
join = os.path.join
#self defined:
from segment_anything import build_sam
from iter_up_refine import IUR
from vertex_filter_connect import SRVFC
from utils.post_process import GetPolygons,generate_coco_ann,transform_polygon_to_original
from utils.polygon import get_candidate_vertex_batch
from eval.iou import IoU
from eval.eval_prompt import PromptMetrics
from utils.losses import BCEDiceLoss

import multiprocessing as mp
import atexit
# import time
import numpy as np
class PromptModel(pl.LightningModule):
    def __init__(self, args,test_cfg=None):
        super().__init__()
        self.args = args
        self.test_cfg=test_cfg
        self.results_poly = []
        self.no_mask_n=0
        if test_cfg.train:
            self.multi_process=False
            load_pl=True if self.args.use_pretrained else False
            self.loss_weight=args.loss_weight
            self.use_cls_model=False #是否使用self.classify_model的flag，训练开始阶段不用
        else:
            self.multi_process=True if not self.args.train_cls else False
            load_pl=True
            self.use_cls_model=True if self.args.train_cls else False
        self.sam_model = build_sam(load_pl=load_pl,**vars(args))
        self.vmap_loss = BCEDiceLoss()#pos_weight=5
        self.bound_loss = BCEDiceLoss(pos_weight=2)        
        self.mask_loss = BCEDiceLoss(pos_weight=2)
        self.iter_decoder = IUR(256, [3, 3, 2])
        if self.args.train_cls:
            self.classify_model=SRVFC(args.large)
            self.classify_loss = F.binary_cross_entropy_with_logits
        
        self.metrics_calculator = PromptMetrics()
        self.avg_process_time=0
        self.avg_pos_process_time=0
        
        if self.multi_process:
            num_processes = 6  # Set the number of processes
            self.pool = mp.Pool(processes=num_processes)        
            atexit.register(self.pool.close)
        
    def forward_step(self, batch,seg_size):
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=batch.get('points',None),
                boxes=batch.get('bbox',None),
                masks=batch.get('mask_prompt',None),
            )
        image_embedding = self.sam_model.image_encoder(batch['img'])

        seg_prob, iou_predictions,pred_poly= self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        map=torch.cat((pred_poly['vmap'],seg_prob),dim=1)#vmap,mask拼接
        seg_prob=F.interpolate(seg_prob, size=seg_size,
            mode='bilinear', align_corners=False)
        iter_res=self.iter_decoder(pred_poly['img_feat'],batch['img'],map)
        res=dict(seg=seg_prob,poly=pred_poly,iter_res=iter_res)#decoder transformer的输出
        return res
    def cal_loss_seg(self,pred_mask,gt_mask,label=''):
        loss_mask = self.loss_weight['mask']*self.mask_loss(pred_mask, gt_mask)
        self.log(f'train/seg_loss{label}', loss_mask, on_step=True, logger=True)
        return loss_mask
    def cal_loss_vertex(self,pred_vmap,gt_vmap,label=''):
        loss_vmap = self.loss_weight['vmap']*self.vmap_loss(pred_vmap, gt_vmap)
        # loss_voff = self.loss_weight['voff']*sigmoid_l1_loss(pred_voff,gt_voff,mask=gt_vmap)
        self.log(f'train/vmap_loss{label}', loss_vmap, on_step=True, logger=True,prog_bar=True)
        # self.log(f'train/voff_loss{label}', loss_voff, on_step=True, logger=True)
        return loss_vmap#+loss_voff
    def cal_loss_edge(self,pred_edge,gt_edge,label=''):
        loss_edge = self.loss_weight['edge']*self.bound_loss(pred_edge, gt_edge)
        self.log(f'train/edge_loss{label}', loss_edge, on_step=True, logger=True)
        return loss_edge
    def on_train_epoch_start(self):
        # 根据当前 epoch 设置 train_cls
        if self.args.train_cls:
            if not self.use_cls_model and self.current_epoch >= self.args.train_cls_start:
                self.use_cls_model = True
                print(f"Start train cls on epoch {self.current_epoch}")        
        # 记录学习率到 TensorBoard
        lr = self.trainer.optimizers[0].param_groups[1]['lr']
        print(f'lr: {lr}')
        self.log('iter_dec_lr', lr, on_step=False, on_epoch=True)
    def training_step(self, batch, batch_idx):
        # gt_mask=batch['gt_mask']
        res=self.forward_step(batch,seg_size=batch['img'].shape[-2:])
        pred_poly,seg_logit=res['poly'],res['seg']
        pred_vmap=pred_poly['vmap']
        iter_res=res['iter_res']
        loss=0
        for scale,name in zip([4,2,1],['s4','s2','s1']):
            gt_mask=batch[name]['gt_mask']
            gt_vmap=batch[name]['vmap']
            pred_maskiter=iter_res[name][:,[1],:,:]
            pred_vmapiter=iter_res[name][:,[0],:,:]
            loss+=self.cal_loss_vertex(pred_vmapiter,gt_vmap,label=f'_{name}')
            loss+=self.cal_loss_seg(pred_maskiter,gt_mask,label=f'_{name}')
            if scale==1:
                loss+=self.cal_loss_seg(seg_logit,gt_mask,label='')
            if scale==4:
                loss+=self.cal_loss_vertex(pred_vmap,gt_vmap,label='')
                pred_maskiter4=F.interpolate(pred_maskiter, scale_factor=4,
                    mode='bilinear', align_corners=False)

        
        if self.use_cls_model:
            vertices=get_candidate_vertex_batch(pred_vmapiter.sigmoid().detach(),scale_size=1)#b,k,2 x,y坐标 原图尺寸坐标 使用s1的vmap结果
            vertex_conf=self.classify_model(vertices,pred_maskiter4.detach())#使用s4 mask resize 4倍作为输入
            #todo 图像特征用哪里的
            cls_gt,weight_mask=self.classify_model.get_target(vertices,batch['polygon'])#polygon 为gt,输入图像尺寸坐标
            pos_weight=1.2 #if self.args.large else 10.0
            loss_classify=self.loss_weight['cls']*self.classify_loss(vertex_conf,cls_gt,weight=weight_mask,pos_weight=torch.tensor([pos_weight],device=vertex_conf.device))

            self.log('train/cls_loss', loss_classify, on_step=True, logger=True,prog_bar=True)
            loss=loss+loss_classify
        if self.args.add_edge:
            loss+=self.cal_loss_edge(pred_poly['edge'],batch['edge'],label='')
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss
    def eval_mask(self,seg_logit,gt_mask,label='',log=True):
        iou_matrix = IoU(seg_logit, gt_mask)#[b,1,h,w]->[b,1]
        miou_mask=sum(iou_matrix)/len(iou_matrix)
        if log:
            self.log(f'val/mIoUmask{label}', miou_mask, on_step=False, on_epoch=True, logger=True,prog_bar=True)

    def validation_step(self, batch, batch_idx,log=True):
        self.ori_size=batch['ori_size']
        # s=time.perf_counter()
        with torch.no_grad():
            res=self.forward_step(batch,seg_size=self.ori_size)
        pred_poly,seg_logit=res['poly'],res['seg']
        pred_maskiter=res['iter_res']['s1'][:,[1],:,:]
        if self.test_cfg.eval:
            gt_mask=batch['s1']['gt_mask']
            pred_maskiter_s2=F.interpolate(res['iter_res']['s2'][:,[1],:,:],
                scale_factor=2, mode='bilinear', align_corners=False)
            pred_maskiter_s4=F.interpolate(res['iter_res']['s4'][:,[1],:,:],
            scale_factor=4, mode='bilinear', align_corners=False)
            self.eval_mask(seg_logit,gt_mask,label='')
            self.eval_mask(pred_maskiter,gt_mask,label='_s1')
            self.eval_mask(pred_maskiter_s2,gt_mask,label='_s2')
            self.eval_mask(pred_maskiter_s4,gt_mask,label='_s4')
        seg_logit=pred_maskiter
        pos_transforms=batch['pos_transform']
        low_iou_num=self.eval_polygon(res['iter_res']['s1'][:,[0],:,:].sigmoid().detach(),seg_logit.detach(),pos_transforms,batch,log,label='')
        
        # end=time.perf_counter()
        # self.avg_process_time+=(end-s)
        return low_iou_num
    def eval_polygon(self,pred_vmap,seg_logit,pos_transforms,batch,log,label=''):        
        ori_img_ids=batch['ori_img_id']
        scale_size=self.ori_size[0]//pred_vmap.shape[-1]
        if self.use_cls_model:
            vertices=get_candidate_vertex_batch(pred_vmap,scale_size)#b,k,2 x,y坐标（输入图像尺寸坐标
            polygons,batch_scores=self.classify_model.predict(vertices,seg_logit)
            batch_polygons,valid_mask,low_iou_num,contour=polygons
        else:
            pool=self.pool if self.multi_process else None
            seg_prob = torch.sigmoid(seg_logit).cpu().numpy().squeeze(1)
            batch_polygons, batch_scores,valid_mask,candidate_vertices,low_iou_num=GetPolygons(
                    seg_prob,pred_vmap,scale_size=scale_size,
                    max_distance=10,pos_transforms=pos_transforms,pool=pool)
        if self.test_cfg.save_results:
            ann_ids=batch['ann_ids']
            category_ids=batch['category_id']
        if self.test_cfg.eval:            
            gt_polygons=batch['polygon']#实例框范围（input_size)内的多边形
        if not np.all(valid_mask):#delete invalid pred instances
            # self.no_mask_n+=no_mask
            valid_idx=np.where(valid_mask)[0]
            print(len(valid_mask)-len(valid_idx),'invalid')
            batch_polygons=[p for p in batch_polygons if p is not None]
            batch_scores=batch_scores[valid_mask]
            if self.test_cfg.eval:
                gt_polygons=[gt_polygons[i] for i in valid_idx]
            pos_transforms=[pos_transforms[i] for i in valid_idx]
            ori_img_ids=[ori_img_ids[i] for i in valid_idx]
            if self.test_cfg.save_results:
                ann_ids=[ann_ids[i] for i in valid_idx]
                category_ids=[category_ids[i] for i in valid_idx]
        else:
            valid_idx=range(len(batch_polygons))
        if self.use_cls_model:
            for b in range(len(batch_polygons)):
                batch_polygons[b]=transform_polygon_to_original(batch_polygons[b], pos_transforms[b])
        for b in range(len(batch_polygons)):
            pred_polygon=batch_polygons[b]
            
            if self.test_cfg.eval:
                gt_polygon=transform_polygon_to_original(gt_polygons[b], pos_transforms[b])
                self.metrics_calculator.calculate_metrics(pred_polygon, gt_polygon)
            if self.test_cfg.save_results:
                ori_img_id=ori_img_ids[b]
                self.results_poly.append(generate_coco_ann(pred_polygon,batch_scores[b],ori_img_id,ann_ids[b],category_ids[b]))
        if log:
            batch_n=len(pred_vmap)
            m=self.metrics_calculator.compute_average(batch_n)
            self.log(f'val/v-precision{label}', m['v_precision'], on_step=False, on_epoch=True, logger=True)
            self.log(f'val/v-recall{label}', m['v_recall'], on_step=False, on_epoch=True, logger=True)
            self.log(f'val/v-f1{label}', m['vf1'], on_step=False, on_epoch=True, logger=True,prog_bar=False)
            self.log(f'val/mIoU{label}', m['miou'], on_step=False, on_epoch=True, logger=True,prog_bar=True)
            self.log(f'val/low_iou_ratio{label}', low_iou_num/batch_n, on_step=False, on_epoch=True, logger=True,prog_bar=True)
        return low_iou_num
    def configure_optimizers(self):
        paramlrs=[]
        if not self.args.freeze_mask:
            paramlrs=[
            {'params': self.sam_model.mask_decoder.parameters(), 'lr': self.args.decoder_lr},
            {'params': self.iter_decoder.parameters(), 'lr': self.args.iter_dec_lr}]
        if not self.args.freeze_img:
            paramlrs.append({'params': self.sam_model.image_encoder.parameters(), 'lr': self.args.img_encoder_lr})
        if self.args.train_cls:
            paramlrs.append({'params': self.classify_model.parameters(), 'lr': self.args.cls_lr})
        optimizer = torch.optim.AdamW(paramlrs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs-self.args.lr_milestone, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[ConstantLR(optimizer, factor=1.0),
                        cosine_scheduler],
            milestones=[self.args.lr_milestone]
        )
        return [optimizer], scheduler
    
    def on_load_checkpoint(self, checkpoint: dict):
        """        
        load_from_checkpoint时，在PromptModel.init build_sam内会插值不对应的权重，
        删除不对应的权重，在之后的on_load_checkpoint不加载这些权重
        """
        # 获取模型的状态字典
        state_dict = checkpoint['state_dict']
        if 'hyper_parameters' in checkpoint and\
        state_dict['sam_model.image_encoder.pos_embed'].shape[1]*16 != checkpoint['hyper_parameters']['args'].image_size:
            # 遍历state_dict中的键，删除指定的键
            keys_to_remove = []
            for key in state_dict.keys():
                if 'image_encoder.pos_embed' in key:
                    keys_to_remove.append(key)
                # 根据你给定的条件删除相关键（比如包含 'rel_pos_h' 或 'rel_pos_w'）
                elif 'rel_pos_h' in key or 'rel_pos_w' in key:
                    keys_to_remove.append(key)
                elif 'output_proj' in key:#large model不用small的顶点分类权重
                    keys_to_remove.append(key)
            
            # 删除这些键
            for key in keys_to_remove:
                print(f"Removing key: {key}")
                del state_dict[key]
            
            # 更新checkpoint中的state_dict
            checkpoint['state_dict'] = state_dict #字典函数内部修改，外部也更新
        return checkpoint
    