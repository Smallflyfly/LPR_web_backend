#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/15 14:38 
"""
import argparse

import cv2
import torch
from PIL import Image
from torchvision import transforms

from STN.model.STN import STNet
from data import cv_imread
from model import build_lprnet
import numpy as np

from test_LPRNet import cv2ImgAddText

CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--image', default='', help='test image')

    args = parser.parse_args()

    return args


def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


transforms = transforms.Compose([
    transforms.Resize([94, 24]),
    transforms.ToTensor()
])


def load_image(image, image_size):
    image = cv_imread(image)
    image = cv2.resize(image, (image_size[0], image_size[1]))
    image = transform(image)
    return image


def show(img, label):
    image = cv_imread(img)
    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(image, lb, (0, 0))
    cv2.imshow("test", img)
    print("predict: ", lb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    args = get_parser()
    LPR_Net = build_lprnet(lpr_max_len=8, phase='test', class_num=len(CHARS))
    LPR_Net = LPR_Net.cuda()
    LPR_Net.load_state_dict(torch.load(args.pretrained_model))
    LPR_Net.eval()
    STN = STNet()
    STN = STN.cuda()
    STN.eval()
    STN.load_state_dict(torch.load('STN/weights/STN_Model_LJK_CA_XZH.pth', map_location=lambda storage, loc: storage))
    print("Successful to build network!")

    image = load_image(args.image, args.img_size)
    image = torch.from_numpy(image)
    image = image.cuda()
    image = image.unsqueeze(0)
    stn_im = STN(image)

    prebs = LPR_Net(stn_im)
    # greedy decode
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    print(preb_labels)
    show(args.image, preb_labels[0])


if __name__ == '__main__':
    test()