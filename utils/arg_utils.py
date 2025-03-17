import os,json
def load_args(parser,path=None):
    args = parser.parse_args()
    if path is None:
        path=f'{args.work_dir}/{args.task_name}/args.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key in vars(args) and key not in ['checkpoint', 'gpus', 'ann_file']:#, 'dataset'
                    setattr(args, key, value) #以上key的值不从args.json中读取
    return args
def load_train_args(parser):
    args = parser.parse_args()
    if 'config' in args:
        with open(args.config, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key in vars(args):
                    setattr(args, key, value)
    return args
