import numpy as np
import pandas as pd
import cv2


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序为RGB
    return image


# 将图片编码成rle格式
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # print(runs)
    # 确保 runs 数组的长度为偶数
    if len(runs) % 2 != 0:
        runs = runs[:-1]  # 如果是奇数长度，去掉最后一个元素

    runs[1::2] -= runs[::2]
    # print(runs)
    return ' '.join(str(x) for x in runs)


# 将rle格式解码成图片
def rle_decode(mask_rle, shape=(512, 512)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    mask_rle = str(mask_rle)
    mask_rle = mask_rle.replace('nan', '')  # 将 'nan' 替换为空字符串
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def load_mask(image_shape, rle_mask):
    mask = rle_decode(rle_mask, image_shape)
    return mask


if __name__ == '__main__':
    train_mask = pd.read_csv('./tcdata/train_mask.csv',
                             sep='\t',
                             names=['name', 'mask'])

    img = cv2.imread('./tcdata/train/' + train_mask['name'].iloc[0])
    print(img.shape)
    mask = rle_decode(train_mask['mask'].iloc[0])
    print(mask.shape)

    print(rle_encode(mask) == train_mask['mask'].iloc[0])
