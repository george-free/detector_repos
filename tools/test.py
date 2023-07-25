from loguru import logger
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Sequence
from exps.default.yolox_tiny import Exp
import time
import torchvision
import cv2
import numpy as np

from yolox.data.data_augment import ValTransform


class Detector:
    def __init__(self,
                 model_path="pre_trained_models/barrel_detect.pth",
                 device="gpu") -> None:
        self.exp = Exp()
        self.device = device
        self.model = self.exp.get_model()
        self.test_size = self.exp.test_size
        self.print_model_info(self.model, self.test_size)
        
        if self.device == "gpu":
            self.model.cuda()
        self.model.eval()
        
        ckpt_file = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt_file["model"])
        
        self.preproc = ValTransform(legacy=False)
        self.fp16 = False
        
        self.num_classes = self.exp.num_classes
        self.confthre = 0.1
        self.nmsthre = self.exp.nmsthre

    def print_model_info(self, model: nn.Module, tsize: Sequence[int]) -> str:
        from thop import profile

        stride = 64
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
        info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
        print("Model Summary: {}".format(info))
        
    def inference(self, img, score_thresh=0.5):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["bboxes"] = []

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = self.postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        output = outputs[0]
        if output is not None:
            output = output.cpu()
            bboxes = output[:, 0:4]
            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for box, cls_id, score in zip(bboxes, cls, scores):
                if score > score_thresh:
                    img_info["bboxes"].append([box, cls_id, score])

        return img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = self.vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
    
    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output
    
    def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img
    
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.000, 1.000, 0.500,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


if __name__ == "__main__":
    detector = Detector()
    image_path = "/home/lutao/data/barrel_datas/images/valid/origins/39900_May051684459320.jpg"
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (0, 0), fx=1/4,
    #                                fy=1/4, interpolation=cv2.INTER_LINEAR)
    detected_infos = detector.inference(image)
    bboxes = detected_infos["bboxes"]
    for bbox in bboxes:
        coords = bbox[0]
        cv2.rectangle(image, 
                      (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])),
                      (0, 0, 255), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)