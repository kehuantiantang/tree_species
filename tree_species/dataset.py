# coding=utf-8
# @Project  ：pytorch-classification 
# @FileName ：dataset.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/5 11:02
import pprint
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import numpy as np

class TreeSpeciesDataset(Dataset):
    def __init__(self, root_path, split_file, class2index, transforms=None):
        self.root_path = root_path
        self.class2index = class2index
        self.transforms = transforms
        self.files = self.get_img_path(split_file)

    def get_img_path(self, split_file):
        counter = defaultdict(lambda : 0)
        items = []
        with open(split_file, 'r') as f:
            files = f.readlines()
            # index, raw_filename,  species,  stage.jpg
            for file in files:
                index, raw_filename, *_, species, stage = file.split('-')
                stage = stage.split('.')[0]
                counter[species] += 1
                items.append((osp.join(self.root_path, file.strip()), index, raw_filename, species, stage))

        pprint.pprint(counter, indent=4)
        return items

    def __getitem__(self, index):
        path, index, raw_filename, *_, species, stage = self.files[index]
        stage = stage.split('.')[0]

        img = Image.open(path)
        img = np.array(img)

        if self.transforms:
            img = self.transforms(image = img)['image']

        return img, self.class2index.index(species), species, stage

    def __len__(self) -> int:
        return len(self.files)


class TreeSpeciesStageDataset(TreeSpeciesDataset):

    def get_img_path(self, split_file):
        counter = defaultdict(lambda : 0)
        items = []
        with open(split_file, 'r') as f:
            files = f.readlines()
            # index, raw_filename,  species,  stage.jpg
            for file in files:
                index, raw_filename, *_, species, stage = file.split('-')
                stage = stage.split('.')[0]

                species = '%s_%s'%(species, stage)
                if species in self.class2index:
                    counter[species] += 1
                    items.append((osp.join(self.root_path, file.strip()), index, raw_filename, species, stage))

        pprint.pprint(counter, indent=4)
        return items