from __future__ import print_function

import os
import sys

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from app.prediction.data import cfg_mnet, cfg_re50
from app.prediction.layers.functions.prior_box import PriorBox
from app.prediction.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import base64
from app.prediction.models.retinaface import RetinaFace
from app.prediction.utils.box_utils import decode, decode_landm
import time

from torchvision import transforms
from PIL import Image


class FaceDetector:
    def __init__(self, network='resnet50', trained_model='./app/prediction/weights/Resnet50_Final.pth', cpu=False):
        torch.set_grad_enabled(False)

        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = self.load_model(self.net, trained_model, cpu)
        self.net.eval()
        print('Finished loading model!')

        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)
        self.resize = 1

        self.confidence_threshold = 0.7
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_thres = 0.7

        # Applying Transforms to the Data
        image_transforms = { 
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                #transforms.RandomRotation(degrees=15),
                transforms.Resize(size=(224, 224)),
                #transforms.RandomHorizontalFlip(),
                #transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=(224, 224)),
                #transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(224, 224)),
                #transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        self.transform = image_transforms['test']

        # model = torch.load('COVID19_resnet101A_19.pt')
        self.model = torch.load('./app/prediction/COVID19-v2_resnet152_9.pt')

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect_faces(self, image_data):
        encoded_data = image_data.split(',')[1]

        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        print('DETECTING FACES...')


        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # print(len(dets))

        faces = []

        print('CLASIFING FACES...')

        for i, b in enumerate(dets):
            conf = b[4]
            # print(conf, args.vis_thres)

            if b[4] < self.vis_thres:
                continue

            b = list(map(int, b))

            face = None

            if b[0] < 0:
                b[0] = 0

            if b[1] < 0:
                b[1] = 0

            face = img_raw[b[1]:b[3], b[0]:b[2]]
            face = Image.fromarray(face)

            # face.save('img'+str(i)+'.png')

            face_tensor = self.transform(face)
            face_tensor = face_tensor.view(1, 3, 224, 224).cuda()
            with torch.no_grad():
                self.model.eval()

                out = self.model(face_tensor)

                # print('out', out)
                _, preds = torch.max(out, 1)

                label = -1

                if preds == 0:
                    label = 0
                else:
                    label = 1

                faces.append({
                    'conf': str(conf),
                    'label': str(label),
                    'x1': str(b[0]),
                    'y1': str(b[1]),
                    'x2': str(b[2]),
                    'y2': str(b[3]),
                })

                # to_write = [img_name, str(conf), str(label), str(int(b[0])), str(int(b[1])), str(int(b[2])), str(int(b[3]))]

                # annotations_file.write(" ".join(to_write) + "\n")

        return faces