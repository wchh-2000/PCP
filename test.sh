# /opt/conda/envs/pcp/bin/python utils/vectorize_res.py
RUN="/opt/conda/envs/pcp/bin/python test.py"
# #loveda训练结果:
$RUN --mode test --task_name small --work_dir 'work_dir_loveda'\
    --ann_file res/seg_small.json
$RUN --mode test --task_name large --work_dir 'work_dir_loveda'\
    --ann_file res/seg_large.json
/opt/conda/envs/pcp/bin/python utils/merge_seg_prompt.py
# 验证精度：
/opt/conda/envs/pcp/bin/python eval/eval_whole.py
