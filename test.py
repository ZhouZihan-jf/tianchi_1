import pandas as pd
import torch
import torchvision.transforms as T
from arguments import args
import model
import cv2
from rle import rle_decode, rle_encode
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(args.img_size),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

# load model
fcn = model.get_model()
fcn.load_state_dict(torch.load(args.resume + 'model_best.pth'))
fcn.to(device)
fcn.eval()


def predict(subm):
    test_mask = pd.read_csv(args.dir + 'test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: args.test_data + x)

    for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
        image = cv2.imread(name)
        image = trfm(image)
        with torch.no_grad():
            image = image.to(device)[None]
            score = fcn(image)['out'][0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])

    subm = pd.DataFrame(subm)
    subm.to_csv(args.result + '/tmp.csv', index=None, header=None, sep='\t')


if __name__ == '__main__':
    subm = []
    predict(subm)
