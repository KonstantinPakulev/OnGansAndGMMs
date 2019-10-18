import os
import pandas
from functools import partial
from skimage import io

from torch.utils.data import Dataset

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

SPLIT_MAP = {
    TRAIN: 0,
    VAL: 1,
    TEST: 2
}


CELEBA_DIR = 'img_align_celeba'


class CelebaDataset(Dataset):

    def __init__(self, dataset_root, mode, transforms):
        super(CelebaDataset, self).__init__()

        self.dataset_root = dataset_root
        self.transforms = transforms

        fn = partial(os.path.join, dataset_root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)

        split = SPLIT_MAP[mode]

        mask = splits[1] == split

        self.filenames = splits[mask].index.values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, id):
        image = io.imread(os.path.join(self.dataset_root, CELEBA_DIR, self.filenames[id]))

        if self.transforms is not None:
            image = self.transforms(image)

        return image


class FlattenTransform:

    def __call__(self, img):
        return img.view(-1)
