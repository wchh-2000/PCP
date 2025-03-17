source activate pcp
# python utils/vectorize_res.py
RUN="python test.py"
# #loveda训练结果:
$RUN --mode test --task_name small --work_dir 'work_dir_loveda'\
    --ann_file res/seg_small.json
$RUN --mode test --task_name large --work_dir 'work_dir_loveda'\
    --ann_file res/seg_large.json
python utils/merge_seg_prompt.py
# 验证精度：
python eval/eval_whole.py
