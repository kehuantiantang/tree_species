# coding=utf-8
import argparse
import json
import pprint
import shutil
import sys
import os
import time
import zipfile
from collections import defaultdict, OrderedDict
import hdfdict as h5d
import logging
import yaml

from utils.logger import Logger


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_current_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr


def load_json(fname):
    with open(fname, "r", encoding= 'utf8') as json_file:
        d = json.load(json_file)
        return d


def save_json(fname, data):
    with open(fname, "w", encoding = 'utf8') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True, ensure_ascii=False)


import pickle
def save_pkl(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding='utf8')



def get_label_name_map():
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'disease': 1,
        'neg':2
    }

    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    print('Label dict:')
    pprint.pprint(reverse_dict)
    return reverse_dict

def filterByconfidence(detpath, threshold, filter_class = None):
    label_dict = get_label_name_map()
    if filter_class is None:
        filter_class = list(label_dict.keys())

    for cls_index in filter_class:
        if cls_index == 0:
            continue
        detfile = os.path.join(detpath, "det_" + label_dict[cls_index] + ".txt")
        new_detfile = []
        with open(detfile, 'r') as f:
            # [img_name, score, xmin, ymin, xmax, ymax]
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in
                          lines]

            for splitline in splitlines:
                if float(splitline[1]) >  threshold:
                    new_detfile.append(splitline)

        # rewrite to det_cls.txt file
        with open(detfile, 'w') as f:
            for a_det in new_detfile:
                f.write(' '.join(a_det) + '\n')

def save_h5(fname, data):
    h5d.dump(data, fname)

def load_h5(path, key):
    array = None
    try:
        res = h5d.load(path, lazy = False)
        if isinstance(key, list):
            return [res[k] for k in key]
        array = res[key]
    except Exception as e:
        print('\nERROR:', path, e)
        sys.exit(-1)
    if array is None:
        print('\nERROR:', path)
        sys.exit(-1)
    return array


import sys
sys.path.append('./')
sys.path.append('../')
import os.path as osp
from utils import Reader
def bbox_counter(path):
    counter = defaultdict(int)
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('xml'):

                objs = Reader(osp.join(root, filename)).get_objects()
                for label in objs['category_name']:
                    counter[label] += 1
    print(counter)


def beauty_argparse(args, verbose = True):
    import prettytable as pt
    if not isinstance(args, dict):
        args = vars(args)

    tb = pt.PrettyTable()
    for key in sorted(args.keys()):
        tb.field_names = ["Name", "Params"]
        tb.add_row([key, args[key]])
    tb.align = 'l'
    if verbose:
        print(tb)
        print()
    return tb.get_string()

def get_format_time(timezone_str = 'Asia/Seoul', time_format = '%Y%m%d-%H%M%S'):
    from pytz import timezone
    from datetime import datetime
    KST = timezone(timezone_str)
    return datetime.now(tz = KST).strftime(time_format)

def save_hyperparams(path, context):
    with open(osp.join(path, 'hyper_params.txt'), 'a+') as file:
        file.write("%s\n"%get_format_time(time_format='%Y-%m-%d %H:%M:%S'))
        file.write(context)
        file.write('%s%s'%('='*80, '\n'))
    return osp.join(path, 'hyper_params.txt')

def hyperparams2yaml(path, context, backup_file = True):

    params = vars(context)
    os.makedirs(osp.join(path, 'yaml'), exist_ok=True)
    f_time = get_format_time()
    with open(osp.join(path, 'yaml', '%s.yaml'%f_time), 'w',
            encoding="UTF-8") as f:
        yaml.dump(params, f, sort_keys=False, allow_unicode = True, indent =4)

    if backup_file:
        backup(osp.join(path, 'yaml', '%s.zip'%f_time))
    Logger.info('Backup:', osp.join(path, 'yaml', '%s'%f_time), '.'*50)
    return osp.join(path, 'yaml', '%s'%f_time)

def yaml2hyperparams(args):
    if osp.exists(args.config):
        opt = vars(args)
        args = yaml.load(open(args.config), Loader = yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    return args

def backup(path, base_path = None, suffixs = ['py', 'yaml', 'sh', 'cfg'], exclude_folder_names = ['output', '.idea']):
    base_path = osp.join(osp.dirname(osp.abspath(__file__)), '../') if base_path is None else base_path

    zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
    for root, _, filenames in os.walk(base_path):
        r = osp.abspath(root)
        if sum([e in r for e in exclude_folder_names]) == 0:
            for filename in filenames:
                if filename.split('.')[-1] in suffixs:
                    zipf.write(os.path.join(root, filename),
                               os.path.relpath(os.path.join(root, filename),
                                               os.path.join(base_path, '..')))
    zipf.close()




if __name__ == '__main__':
    backup('/home/khtt/code/pytorch-classification/rhythm_segmentation/output/unknown/test/a.zip')