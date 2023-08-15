import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
from arguments import args
import cv2
from rle import rle_decode, rle_encode
import torchvision.transforms as T
import torch.utils.data as D

trfm = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])


class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(args.img_size),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


train_mask = pd.read_csv(args.dir + 'train_mask.csv', sep='\t', names=['name', 'mask'])
train_mask['name'] = train_mask['name'].apply(lambda x: args.train_data + x)

dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm, False
)

valid_idx, train_idx = [], []
for i in range(len(dataset)):
    if i % 7 == 0:
        valid_idx.append(i)
    #     else:
    elif i % 7 == 1:
        train_idx.append(i)

train_ds = D.Subset(dataset, train_idx)
valid_ds = D.Subset(dataset, valid_idx)

# define training and validation data loaders
loader = D.DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

vloader = D.DataLoader(
    valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



