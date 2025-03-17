# SAMPolyBuild
This repository is the code implementation of the paper "PCP: A Prompt-based Cartographic-level Polygonal Vector Extraction Framework for Remote Sensing" submitted to TGRS.

## Installation
Conda virtual environment is recommended for installation. Please choose the appropriate version of torch and torchvision according to your CUDA version.
Run the following commands or run the 'install.sh' script.
```shell
conda create -n pcp python=3.10 -y
source activate pcp # or conda activate pcp
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
Download the SAM vit_b model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in the 'segment_anything' folder.

## Dataset Preparation
First, train a semantic segmentation model (any), and predict the segmentation results.

(If the dataset is in raster format, run 'dataset/vectorize_gt.py' to get the simplified vector annotations in coco format.)

Then, for prompt training, run 'utils/get_train_mask_prompt.py' to match the ground truth (GT) with the segmentation results as the mask prompt (not all GT is matched).
### LoveDA dataset building land vector
The original LoveDA dataset can be download from [here](https://github.com/Junjue-Wang/LoveDA).
We use the validation set for testing because the labels are required for conversion to vector form, whereas the test set labels are unavailable.
Using the SFA-Net to generate the segmentation results, after the steps above, we get the vector annotations in
[coco format](https://pan.baidu.com/s/19K8LeQ5pCyRc7hbWdJszsQ?pwd=p6r3), place them under 'dataset/loveda', combined with the original LoveDA image folder.
The 'ann_build.json' contains simplified building area polygons, and the 'ann_small_prompt.json' and 'ann_large_prompt.json' contain the GT polygons and mask prompt of small and large area, respectively.

## Training
Training the small and large input size models on the LoveDA dataset:
```shell
python train.py --config configs/loveda_small.json
python train.py --config configs/loveda_large.json
```
## Testing
Given the segmentation results, first, vectorize the segmentation results to coco format. For LoveDA dataset the processed results can be downloaded from [here](https://pan.baidu.com/s/19K8LeQ5pCyRc7hbWdJszsQ?pwd=p6r3) in the 'Val/Rural/res' folder.
```shell
python utils/vectorize_res.py
```
Then, use trained PCP model to predict the polygons. You need to change the **--task_name** to the corresponding training task name, and the other arguments will be set automatically according to training configuration.
The trained model loveda_small.pth and loveda_large.pth can be downloaded from [here](https://pan.baidu.com/s/1bhPH0jBbUiVgTZfct9tCrw?pwd=drpp), and replace the 'args.checkpoint' in 'test.py' with the path of the downloaded model.
```shell
python test.py --task_name small --ann_file res/seg_small.json
python test.py --task_name large --ann_file res/seg_large.json
```

Merge the segmentation results and the predicted polygons to get the final results.
```shell
python utils/merge_seg_prompt.py
```

Evaluate the final results based on C-IoU and vertex metrics.
```shell
python eval/eval_whole.py
```
Visualize the results.
```shell
python show.py
```

## Acknowledgement
This project is developed based on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
 and [SAMPolyBuild](https://github.com/wchh-2000/SAMPolyBuild) project.

## License

This project is licensed under the [Apache 2.0 license](LICENSE).

## Contact
If you have any questions, please contact wangchenhao22@mails.ucas.ac.cn.
