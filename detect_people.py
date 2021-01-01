# python detect_people.py --weights WEIGHTS --cfg CONFIG_FILE\
# --det SAVE_FOLDER --img IMAGE_FILE


from __future__ import division
import time
import torch
# import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import write_results, load_classes
from darknet import Darknet
# from preprocess import inp_to_image  # , prep_image
# import pandas as pd
import random
import argparse
import pickle as pkl
import math


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]  # 1->width, 0->hight
    # サイズ変更しているのは処理用のimg配列
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
    personCheck = 0
    square = 0

    # c1 = tuple(x[1:3].int())  # 一つ目の端点
    # 座標取り出し
    x1 = x[1].int().item()
    y1 = x[2].int().item()

    # c2 = tuple(x[3:5].int())  # 二つ目の端点
    # 座標取り出し
    x2 = x[3].int().item()
    y2 = x[4].int().item()

    try:
        # バグポイント0
        # if(math.isnan(x[-1]) or x[1] == 0 or x[2] == 0 or x[3] == 0 or x[4] == 0):
        # if(math.isnan(x[-1]) and x[1] == 0 and x[2] == 0 and x[3] == 0 and x[4] == 0):
        if math.isnan(x[-1]) or (x[-1] < 0 or 80 < x[-1]):
            return [[np.nan, np.nan], [np.nan, np.nan], personCheck, square]
        else:
            cls = int(x[-1])
    except Exception as e:
        print(e)
    # バグポイント1
    # label = "{0}".format(classes[cls])
    '''
    try:
        label = "{0}".format(classes[cls])
    except IndexError as e:
        print(x)
    '''
    if cls == 0:  # True(All) or cls == 0(Drow only person)
        widX = x2 - x1
        heiY = x2 - x1
        square = widX * heiY
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        # c3 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        # ラベルの書き込み, -1は塗りつぶしの引数（負だと？）
        # cv2.rectangle(img, c1, c3, color, -1)
        # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
        #             cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        personCheck = 1
    return [[x1, y1], [x2, y2], personCheck, square]
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
    # 群衆のため 0.1 -> 0.4
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help="Input resolution of \
                                                    the network. Increase to \
                                                    increase accuracy. \
                                                    Decrease to \
                                                    increase speed",
                        default="608", type=str)
    # 追加のarg
    parser.add_argument("--weights", dest="wei",
                        help="Designate trained weights",
                        default="yolov3-tiny-obj-add_final.weights")
    parser.add_argument("--cfg", help="Setting network configuration",
                        default="yolov3-tiny-obj-add.cfg")
    parser.add_argument("--img", help="Designated image file",
                        default=None)
    parser.add_argument("--det", help="Location that is saved image",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    cfgfile = args.cfg  # いじったcfgファイルの指定
    # 注: あとで設定
    weightsfile = args.wei  # weightsの指定
    # num_classes = 1  # クラス数, 今回はPeopleのみ
    num_classes = 80

    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0  # fps計算に使用
    CUDA = torch.cuda.is_available()

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

    personCount = 0
    start = time.time()

    # video or cam -> frame
    # image -> image
    # ret, frame = cap.read()
    image = cv2.imread(args.img)

    # tensorFrame, frame, origin size
    img, orig_im, dim = prep_image(image, inp_dim)

    # print(orig_im.shape)  # ex.(720, 1280, 3)

    # im_dim = torch.FloatTensor(dim).repeat(1,2)
    # cuda入ってないから無視
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    # ここでNNに通す
    # util.pyに存在
    output = write_results(output, confidence, num_classes, nms=True,
                           nms_conf=nms_thresh)

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
    output[:, [1, 3]] *= image.shape[1]  # width
    output[:, [2, 4]] *= image.shape[0]  # hight

    classes = load_classes('data/coco.names')
    # classes = load_classes('data/obj.names')
    colors = pkl.load(open("pallete", "rb"))

    # lambdaとmapとlistの組み合わせ, xはoutputの要素を参照
    # 初期バグ位置
    # p_list = list(map(lambda x: write(x, orig_im), output))
    p_list = list(map(lambda x: write(x, orig_im), output))
    # スライスの変換にはndarrayじゃないとダメ
    # 構造を整形しておかないと返還後のスライスでエラーが出る
    p_list = np.array(p_list)
    # print(p_list)
    # for i in range(0, len(p_list)):
    #     print(p_list[i].shape)
    personCheck = p_list[:, 2]
    square = p_list[:, 3]
    p_list = p_list[:, 0:2]
    # print(type(p_list[0][0][0]))  # int
    '''
    # 座標情報引き渡しのテスト, p_list[0][0][0]を見れば検知されたか確認可能
    # print(len(p_list))  # -> 人数（0人だと1)
    # print(p_list)
    ----------------
    [[[nan]]]  # 0 exists
    [([160, 22], [202, 122])]  # 1 person
    [([252, 188], [297, 283]), ([188, 5], [235, 98]), \
    ([118, 27], [160, 131])]  # 3 people
    # -> p_listはタプルとして現れる
    # -> 検出がなかったときの処理を書きたいときは
    #    if math.isnan(p_list[0][0][0]):
    # タプルの扱い
    #   p_list[n-1][0] -> n番目の矩形の左上座標[x, y]
    #   p_list[n-1][1] -> n番目の矩形の右下座標[x, y]
    # バグ修正のため, 座標点に0があるとnanにしていた
    # -> p_list[n-1][0][0]がmath.isnanでTrueだと
    #    その座標を採用しないとすると良さそう
    ----------------
    if not math.isnan(p_list[0][0][0]):
        print(p_list[0][0])
        print(p_list[0][0][0])
        a = p_list[0][0][0]
        print(a)
    '''

    # 人数カウントの処理
    # print(len(output))  # 全てのクラスの検出数
    squareAv = int(square[square.nonzero()].mean())
    for i in range(0, len(output)):
        if math.isnan(personCheck[i]):
            continue
        if personCheck[i] == 1:
            if square[i] < (4 * squareAv):
                personCount = personCount + 1
                color = random.choice(colors)
                cv2.rectangle(orig_im, (p_list[i][0][0], p_list[i][0][1]),
                              (p_list[i][1][0], p_list[i][1][1]), color, 2)
    print("FPS of the video is {:5.2f}".format(1 / (time.time() - start)))
    print("{} people have detected!!".format(personCount))
    # 文字書き込み
    text = "{} people have detected.".format(personCount)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    corner = t_size[0] + 1, t_size[1] + 1
    # print(type(corner))  # tupple
    # ラベルの書き込み, -1は塗りつぶしの引数（負だと？）
    cv2.rectangle(orig_im, (0, 0), corner, (255, 255, 255), -1)
    cv2.putText(orig_im, text, (0, t_size[1]),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    imgName = args.img.split("/")
    # print(imgName[-1])  # image file name
    det = args.det
    det = det + "/Processed_" + imgName[-1]
    if det is not None:
        cv2.imwrite(det, orig_im)

    cv2.imshow("Processed Image", orig_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print("0: " + str(type(output)))  # typeはtorch.Tensor
    # print("0: " + str(output.shape))  # 8*検出数の構造の配列
