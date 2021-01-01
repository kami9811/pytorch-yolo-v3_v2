from __future__ import division
import time
import torch
# import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import write_results, load_classes
from darknet import Darknet
# from preprocess import prep_image, inp_to_image
# import pandas as pd
import random
import argparse
import pickle as pkl
import math
import asyncio


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]  # 1->width, 0->hight
    img = cv2.resize(orig_im, (inp_dim, inp_dim))  # 元の画像をreso指定に
    # ::-1 -> 後ろから1個ずつ参照（最初）:（最後）:（ステップ幅）, BGR->RGB
    # transpose h, w, c -> c, h, w
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    '''
    # 使用例↓
    # lambdaとmapとlistの組み合わせ, xはoutputの要素を参照
    list(map(lambda x: write(x, orig_im), output))
    '''
    # print(x)
    '''
    # 何も読み込まれなかった時の座標
    # 1,2->端点1, 3,4->端点2(対角), -1->クラス
    tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.7417e-24, 1.4013e-45,
               nan])
    '''
    c1 = tuple(x[1:3].int())  # 一つ目の端点
    # 座標取り出し
    x1 = x[1].int().item()
    y1 = x[2].int().item()

    c2 = tuple(x[3:5].int())  # 二つ目の端点
    # 座標取り出し
    x2 = x[3].int().item()
    y2 = x[4].int().item()
    # .item()をつけると <class 'int'>
    # .item()をつけないと <class 'torch.Tensor'>
    # print(type(y2))
    # print("x1: {0} {1}, y1: {2} {3}, x2: {4} {5}, y2: {6} {7}".format(x1, x11, y1, y11, x2, x21, y2, y21))

    # バグポイント0
    if(math.isnan(x[-1]) or x[1] == 0 or x[2] == 0 or x[3] == 0 or x[4] == 0):
        return [[np.nan]]
    else:
        cls = int(x[-1])
    # バグポイント1
    label = "{0}".format(classes[cls])
    '''
    try:
        label = "{0}".format(classes[cls])
    except IndexError as e:
        print(x)
    '''
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c3 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # ラベルの書き込み, -1は塗りつぶしの引数（負だと？）
    cv2.rectangle(img, c1, c3, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return [x1, y1], [x2, y2]
    '''
    # Origin script
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # ラベルの書き込み, -1は塗りつぶしの引数（負だと？）
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img
    '''


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.25)
    '''
    # 2重検知が多いため数値変更
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    '''
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.1)
    parser.add_argument("--reso", dest='reso', help="Input resolution of \
                                                    the network. Increase to \
                                                    increase accuracy. \
                                                    Decrease to \
                                                    increase speed",
                        default="160", type=str)
    # 追加のarg
    parser.add_argument("--weights", dest="wei",
                        help="Designate trained weights",
                        default="../yolov3-tiny-obj-add_final.weights")
    return parser.parse_args()


async def main():
    for i in range(40):
        ret, frame = cap.read()
        if ret:
            # tensorFrame, frame, origin size
            img, orig_im, dim = prep_image(frame, inp_dim)

            # print(orig_im.shape)  # ex.(720, 1280, 3)

            #            im_dim = torch.FloatTensor(dim).repeat(1,2)
            # cuda入ってないから無視
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            # ここでNNに通す
            output = write_results(output, confidence, num_classes, nms=True,
                                   nms_conf=nms_thesh)

            # 全然引っかからないからとりあえず無視？
            '''
            if type(output) == int:
                frames += 1
                # 今どれくらいのフレームで動いているかの出力
                print("FPS of the video is {:5.2f}"
                      .format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                # ord(q)->qのアスキー変換, 動作的にはqを押したら終了
                if key & 0xFF == ord('q'):
                    break
                continue
            '''

            # 上と下の制限
            # 0~1で正規化, 分解能はreso（defaultでは160）
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim))/inp_dim
            # im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]  # width
            output[:, [2, 4]] *= frame.shape[0]  # hight

            # classes = load_classes('data/coco.names')
            # classes = load_classes('../data/obj.names')
            # colors = pkl.load(open("pallete", "rb"))

            # lambdaとmapとlistの組み合わせ, xはoutputの要素を参照
            # 初期バグ位置
            p_list = list(map(lambda x: write(x, orig_im), output))
            '''
            # 座標情報引き渡しのテスト, p_list[0][0][0]を見れば検知されたか確認可能
            print(p_list)
            if not math.isnan(p_list[0][0][0]):
                print(p_list[0][0])
                print(p_list[0][0][0])
                a = p_list[0][0][0]
                print(a)
            '''
            print(len(output))  # bbの数

            cv2.imshow("frame", orig_im)
            # print("0: " + str(type(output)))  # typeはtorch.Tensor
            # print("0: " + str(output.shape))  # 8*検出数の構造の配列
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                # break
                global continueFlag
                continueFlag = 0
            global frames
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break


if __name__ == '__main__':
    # ループ前のYoloネットワーク読み込みなど初期設定
    cfgfile = "../cfg/yolov3-tiny-obj.cfg"  # いじったcfgファイルの指定
    # 注: あとで設定
    weightsfile = "../yolov3-tiny-obj-add_final.weights"  # weightsの指定
    num_classes = 1  # クラス数, 今回はPersonのみ
    '''
    cfgfile = "cfg/yolov3-tiny.cfg"
    weightsfile = "yolov3-tiny.weights"
    num_classes = 80
    '''
    classes = load_classes('../data/obj.names')
    colors = pkl.load(open("pallete", "rb"))

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    weightsfile = args.wei
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 1
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    videofile = 'video.avi'

    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()

    continueFlag = 1
    # 初期設定終わりでループパートへ
    # カメラとリンクしているなら動作
    while cap.isOpened() and continueFlag == 1:
        loop = asyncio.get_event_loop()
        tasks = asyncio.gather(
            main(),
        )
        # 非同期実行、それぞれが終わるまで
        results = loop.run_until_complete(tasks)
        print(results)
        # 非同期実行
        # results = loop.run_until_complete(main())
