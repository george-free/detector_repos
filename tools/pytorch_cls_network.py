import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import json
from glob import glob
from functools import wraps
import cv2
import random
import numpy as np
import copy
from torchsummary import summary
import sys
from torchviz import make_dot


class BottleNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self._leftpath = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels= out_channels // 2,
                kernel_size=1,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.PReLU(),
            nn.BatchNorm2d(out_channels // 2),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self._rightpath = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False
            ),
            nn.PReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.PReLU(),
            nn.BatchNorm2d(out_channels // 2),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        x_bottle_left = self._leftpath(x)
        x_bottle_right = self._rightpath(x)
        return torch.cat((x_bottle_left, x_bottle_right), dim=1)
    

class DataSet(torchDataset):
    def __init__(self,
                 data_dir,
                 mean_data=None,
                 std_data=None,
                 img_size=(112, 112),
                 step="train") -> None:
        if not os.path.exists(data_dir):
            print("data dir - {} does not exist".format(data_dir))
        self.__input_dim = img_size  # height, width
        self._cats = os.listdir(data_dir)
        # self._class_ids = [(index, self._cats) for index in range(0, len(self._cats))]
        self._class_ids = {self._cats[index]: index for index in range(0, len(self._cats))}
        self._num_class = len(self._cats)
        self._data_dir = data_dir
        self._image_ids = []
        
        image_id = 1
        for label in self._cats:
            for image_path in glob(os.path.join(data_dir, label) + "/*.jpg"):
                self._image_ids.append([image_id, image_path, label])
                image_id += 1
        # shuffle the data
        random.shuffle(self._image_ids)
        if mean_data is None or std_data is None:
            self._image_mean, self._image_std = self.cal_mean_std(self._image_ids)
        else:
            self._image_mean = mean_data
            self._image_std = std_data
        self._num_images = len(self._image_ids)
        
    @property
    def num_classes(self):
        return self._num_class
    
    @property
    def input_dim(self):
        return self.__input_dim
    
    @classmethod
    def preproc(cls,
                img,
                input_dim,
                mean,
                std,
                swap=(2, 0, 1), padding:int=114):
        assert len(img.shape) == 3
        padded_resized_image = cls.resize_image(img, input_dim, padding)
        
        # normalzied
        for channel in range(0, 3):
            padded_resized_image[channel] = (padded_resized_image[channel] - mean[channel]) / std[channel]
        padded_resized_image = padded_resized_image.transpose(swap)
        
        return padded_resized_image
    
    def __len__(self):
        return self._num_images
        
    def cal_mean_std(self, data_list):
        means = []
        variances = []
        for data in data_list:
            _, image_path, _ = data
            # print("img path: {}".format(image_path))
            img = cv2.imread(image_path)
            img = cv2.resize(
                img,
                (self.__input_dim[1], self.__input_dim[0]),
                interpolation=cv2.INTER_LINEAR)
            # img = img.transpose((2, 0, 1))
            # print(img.shape)
            means.append(np.mean(img, axis=(0, 1)))
        means = np.array(means, dtype=np.float32)
        # print("means shape: {}".format(means.shape))
        mean_bgr = np.mean(means, axis=0)
        for data in data_list:
            _, image_path, _ = data
            img = cv2.imread(image_path)
            img = cv2.resize(
                img,
                (self.__input_dim[1], self.__input_dim[0]),
                interpolation=cv2.INTER_LINEAR)
            # img = img.transpose((2, 0, 1))
            img_var = np.mean((img - mean_bgr) ** 2, axis=(0, 1))
            variances.append(img_var)

        std_bgr = np.sqrt(np.mean(variances, axis=0))
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
        std_rgb = np.array([std_bgr[2], std_bgr[1], std_bgr[0]])
        
        print("mean: {}".format(mean_rgb))
        print("std: {}".format(std_rgb))
        return mean_rgb, std_rgb
        
    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim
    
    # @staticmethod
    # def mosaic_getitem(getitem_fn):
    #     """
    #     Decorator method that needs to be used around the ``__getitem__`` method. |br|
    #     This decorator enables the closing mosaic

    #     Example:
    #         >>> class CustomSet(ln.data.Dataset):
    #         ...     def __len__(self):
    #         ...         return 10
    #         ...     @ln.data.Dataset.mosaic_getitem
    #         ...     def __getitem__(self, index):
    #         ...         return self.enable_mosaic
    #     """

    #     @wraps(getitem_fn)
    #     def wrapper(self, index):
    #         if not isinstance(index, int):
    #             self.enable_mosaic = index[0]
    #             index = index[1]

    #         ret_val = getitem_fn(self, index)

    #         return ret_val

    #     return wrapper

    def __getitem__(self, index):
        id_ = self._image_ids[index]
        image_id, image_path, label_name = id_
        label_id = self._class_ids[label_name]
        
        img = cv2.imread(image_path)
        img_size = img.shape[0:2]
        assert img is not None, f"file named {image_path} not found"
        preprocessed_img = self.preproc(img, self.__input_dim, self._image_mean, self._image_std)
        return copy.deepcopy(preprocessed_img), copy.deepcopy(label_id)
    
    @classmethod
    def resize_image(cls, img, input_dim, padding=114):
        padded_image = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * padding
        ratio = min(input_dim[0] / img.shape[0],
                    input_dim[1] / img.shape[1])
        try:
            resized_img = cv2.resize(img, 
                                    (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                                    interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            padded_image[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
        except Exception as e:
            print("e - {}, img shape: {}, ratio: {}, rshape: {}".format(
                e, img.shape, ratio, resized_img.shape))
            # cv2.imwrite("debug.png", img)
            raise Exception(e)
        
        return padded_image


class MyNetwork(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 step="train",
                 use_l1=True):
        super().__init__()
        self._num_classes = num_classes
        self._step = step
        self._num_backbone = 1
        output_channels_backbone = [256, 128, 128]
        
        layer_dict = OrderedDict()
        for index in range(0, self._num_backbone):
            bottle_base = BottleNetwork(in_channels, output_channels_backbone[index])
            in_channels = output_channels_backbone[index]
            layer_dict[bottle_base] = f'btn_{index}'
        self._backbone = nn.ModuleList(layer_dict)

        in_channels = output_channels_backbone[0]
        self._cls_convs = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._num_classes,
            kernel_size=5,
            stride=1,
            padding=1,
            dilation=1,
            groups=1
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        x = inputs
        for layer in self._backbone:
            x = layer(x)
        x = self._cls_convs(x)
        
        # max pooling
        x = nn.MaxPool2d(
            kernel_size=(x.shape[2], x.shape[3])
        )(x)
        
        # soft max
        x = nn.Softmax(dim=1)(x)

        return x


class MyTrainer:
    def __init__(self,
                 batch_size,
                 max_epoch,
                 data_dir,
                 basic_learning_rate,
                 model_save_dir,
                 num_classes=None,
                 warmup_epoches=5,
                 warmp_lr=0,
                 use_l1: bool=True,
                 device="cpu") -> None:
        self._use_l1 = use_l1
        self._device = device
        self._batch_size = batch_size
        self._max_epoch = max_epoch
        self._basic_learning_rate = basic_learning_rate
        self._warmup_lr = warmp_lr
        self._warmup_epoches = warmup_epoches
        self._epoch = 0  # nth poch
        self._iter = 0  # nth step in each poch
        self._model_name = "cls_bottleneck"
        
        train_data_dir = os.path.join(data_dir, "train")
        self._train_dataset = DataSet(
            data_dir=train_data_dir,
            step="train"
        )
        self._train_data_loader = DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True)
        self._max_iter = len(self._train_data_loader) # train data loader has considered the factor of batchsize

        valid_data_dir = os.path.join(data_dir, "valid")
        self._valid_dataset = DataSet(
            data_dir=valid_data_dir,
            step="valid"
        )
        self._valid_data_loader = DataLoader(
            self._valid_dataset, batch_size=self._batch_size, shuffle=False
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        if num_classes is None:
            self._num_classes = self._train_dataset.num_classes
        else:
            self._num_classes = num_classes
        print("num classes: {}".format(self._num_classes))
        self._model = MyNetwork(self._num_classes)
        # summary models
        input_size = self._train_dataset.input_dim
        summary(self._model, (3, input_size[0], input_size[1]))
        
        # before train, setup the train env
        self._optimizer = self.get_optimizer(
            self._model,
            self._basic_learning_rate,
            batch_size=self._batch_size)
        self._lr_scheduler = lr_scheduler.StepLR(
            optimizer=self._optimizer, step_size=10, gamma=0.2)
        
        # for save models
        self._best_models = None
        self._best_ap = 0.0
        self._save_model_dir = model_save_dir
        
    def visualize_computation_graph(self):
        x = torch.randn([1, 3, self._train_dataset.input_dim[0], self._train_dataset.input_dim[1]]).to(torch.float32)
        y = self._model(x)
        print("show computation graph")
        dot_graph = make_dot(y, params=dict(self._model.named_parameters()))
        dot_graph.format = "pdf"
        dot_graph.render(filename=f"{self._model_name}_computation_graph")
        
    def train(self):
        # TODO: use scaler for train in the feature
        # set the model to the device
        self._model.to(self._device)
        self._model.train()

        for self._epoch in range(1, self._max_epoch + 1):
            # train
            train_total_loss = 0
            train_total_correct = 0
            train_total_samples = 0
            self._iter = 0
            for train_data in enumerate(self._train_data_loader):
                self._iter = train_data[0]
                inputs, targets = train_data[1]
                # print("index: {}, targets:{}".format(index, targets))
                self._optimizer.zero_grad()
                
                inputs = inputs.to(torch.float32)
                # targets = targets.to(torch.float32)
                # targets = targets.unsqueeze(1)
                # print(targets)
                outputs = self._model(inputs)
                outputs = outputs.squeeze()
                loss = self.loss_fn(self._model, outputs, targets)
                loss.backward()
                self._optimizer.step()
                
                # upgrade learning rate
                new_lr = self._lr_scheduler.update_lr(self._epoch * self._max_iter + self._iter)
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = new_lr
                
                train_total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total_correct += (predicted == targets).sum().item()
                train_total_samples += inputs.size(0)
                print("\rpoch:{}, itr:{}, loss: {:.04f}".format(
                    self._epoch, self._iter, loss.item()), end=' ')
                # sys.stdout.write("loss: {:.04f}".format(loss.item()))
                # sys.stdout.flush()

            print("train epoch {}, avg loss = {:.04f}, acc = {:.04f}".format(
                self._epoch, train_total_loss / train_total_samples, train_total_correct / train_total_samples))

            valid_total_loss = 0
            valid_total_correct = 0
            valid_total_samples = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self._valid_data_loader):
                    inputs = inputs.to(torch.float32)
                    # targets = targets.unsqueeze(1)
                    outputs = self._model(inputs)
                    outputs = outputs.squeeze()
                    loss = self.loss_fn(self._model, outputs, targets)
                    valid_total_loss += loss.item() * inputs.size(0)
                    # _, predicted = torch.max(outputs, 1)
                    predicted = torch.argmax(outputs, dim=1)
                    valid_total_correct += (predicted == targets).sum().item()
                    valid_total_samples += inputs.size(0)
            print("valid epoch {}, avg loss = {:.04f}, acc = {:.04f}".format(
                self._epoch, valid_total_loss / valid_total_samples, valid_total_correct / valid_total_samples
            ))
            if valid_total_correct / valid_total_samples > self._best_ap:
                self._best_ap = valid_total_correct / valid_total_samples
                self._best_models = self._model
                self.save_models()
            
    def resume(self):
        pass
    
    def save_checkpoint(self, state, save_dir, model_name=""):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + "_best_ckpt.pth")
        torch.save(state, filename)
    
    def save_models(self):
        if self._best_models is None:
            print("The model is None, can not save")
        ckpt_state = {
            "start_epoch": self._epoch,
            "model": self._best_models.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "best_ap": self._best_ap,
        }
        self.save_checkpoint(ckpt_state, self._save_model_dir, self._model_name)
    
    def loss_fn(self, model, inputs, labels, use_l1=True):
        loss = nn.CrossEntropyLoss()(inputs, labels)
        if use_l1:
            l1_lamda = 0.01
            l1_norm = torch.norm(torch.cat(
                [parameter.view(-1) for parameter in model.parameters()]), p=1)
            loss += l1_lamda * l1_norm
        
        return loss
    
    def get_optimizer(self, model, basic_lr, batch_size):
        backbone_parameters = []
        cls_conv_parameters = []
        
        # print("named modules: {}".format(model.named_modules()))
        for name, param in model.named_modules():
            if "_backbone" in name and "path." in name and isinstance(param, nn.Conv2d):
                # print("backbone param: {}".format(param.weight))
                backbone_parameters.append(param.weight)
            elif "_cls_convs" in name and isinstance(param, nn.Conv2d):
                # print("cls_convs param: {}".format(param.weight))
                cls_conv_parameters.append(param.weight)

        optimizer = torch.optim.SGD(
            [{"params": backbone_parameters, "weight_decay": 0.00001, "lr": basic_lr * 0.1},
             {"params": cls_conv_parameters, "weigh_decay": 0.00001, "lr": basic_lr * 0.2}],
            lr=basic_lr,
            momentum=0.9,
            nesterov=True
        )
        return optimizer


class MyEvaluater:
    def __init__(self,
                 num_classes,
                 model_path,
                 device="cpu",
                 show_label=False) -> None:
        self._model = MyNetwork(num_classes)
        cpkt = torch.load(model_path)
        self._model.load_state_dict(cpkt["model"])
        self._model.to(device)
        self._model.eval()
        self._show_label = show_label
    
    def evaluate_batches(self, data_dir, batch_size = 4):
        valid_dataset = DataSet(
            data_dir=data_dir,
            step="test"
        )
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        num_datas = len(valid_data_loader)
        corrected = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_data_loader):
                inputs = inputs.to(torch.float32)
                outputs = self._model(inputs)
                outputs = outputs.squeeze()
                predicted = torch.argmax(outputs, dim=1)
                corrected += (predicted == targets).sum().item()
        print("{} datas, {} corrected.".format(num_datas, corrected))
        
    def evaluate(self,
                 image,
                 input_dim=(112, 112),
                 mean=[112.13499, 112.12045, 104.224945],
                 std=[58.99968, 53.3359, 56.122524]):
        img = DataSet.preproc(image, input_dim, mean, std)
        img = img[np.newaxis, ...]
        with torch.no_grad():
            inputs = torch.tensor(img).to(torch.float32)
            outputs = self._model(inputs)
            outputs = outputs.squeeze()
            predicted = torch.argmax(outputs)
            print("predicted: {}".format(predicted))


def train():
    trainer = MyTrainer(
        batch_size=4,
        max_epoch=50,
        data_dir="/home/lutao/datasets/classification/trashcan_datasets",
        basic_learning_rate=0.01,
        model_save_dir="./saved_models",
    )
    trainer.train()
    
def val():
    validater = MyEvaluater(
        num_classes=2,
        model_path="/home/lutao/dev/YOLOX/tools/saved_models/cls_bottleneck_best_ckpt.pth")
    image_dir = "/home/lutao/datasets/classification/trashcan_datasets/test/overflowed/"
    for img_path in glob(image_dir + "*.jpg"):
        image = cv2.imread(img_path)
        validater.evaluate(image)
        cv2.imshow("image", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        else:
            continue

def visualize_graph():
    trainer = MyTrainer(
        batch_size=4,
        max_epoch=50,
        data_dir="/home/lutao/datasets/classification/trashcan_datasets",
        basic_learning_rate=0.01,
        model_save_dir="./saved_models",
    )
    trainer.visualize_computation_graph()

if __name__ == "__main__":
    visualize_graph()
    