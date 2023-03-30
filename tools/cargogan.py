from torch.utils.data.dataset import Dataset as torchDataset
import os
from glob import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

import math
from copy import deepcopy
import xml.etree.ElementTree as ET


class UnetDown(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 normalize=True):
        super(UnetDown, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Conv2d(
        #         in_size,
        #         out_size,
        #         kernel_size=4,
        #         stride=2,
        #         padding=2,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(out_size, eps=0.8),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.5)
        # )
        self.down_conv_layer = nn.Conv2d(
                in_size,
                out_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        self.layer_bach_norm = nn.BatchNorm2d(out_size, eps=0.8)
        self.layer_relu = nn.LeakyReLU(0.2)
        self.layer_dropout = nn.Dropout(0.5)
        
        self.model = nn.ModuleList()
        self.model.append(self.down_conv_layer)
        if normalize:
            self.model.append(self.layer_bach_norm)
        self.model.append(self.layer_relu)
        self.model.append(self.layer_dropout)
    
    def forward(self, x):
        y = x
        for model in self.model:
            y = model(y)
        return y
    
    
class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super(UnetUp, self).__init__()
        # self.model = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_size,
        #         out_size,
        #         kernel_size=4,
        #         stride=2,
        #         padding=2,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(out_size, eps=0.8),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5)
        # )

        self.up_conv_layer = nn.ConvTranspose2d(
                in_size,
                out_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        self.layer_bach_norm = nn.BatchNorm2d(out_size, eps=0.8)
        self.layer_relu = nn.ReLU(inplace=True)
        self.layer_dropout = nn.Dropout(0.5)
        
        self.model = nn.ModuleList()
        self.model.append(self.up_conv_layer)
        if normalize:
            self.model.append(self.layer_bach_norm)
        self.model.append(self.layer_relu)
        self.model.append(self.layer_dropout)

    def forward(self, x, skip_input):
        y = x
        for model in self.model:
            y = model(y)
        # x = self.model(x)
        out = torch.cat((y, skip_input), 1)
        return out
        

class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        
        channels, img_height, img_width = input_shape
        self.down1 = UnetDown(channels, 64, normalize=True)
        self.down2 = UnetDown(64, 128)
        self.down3 = UnetDown(128, 256)
        self.down4 = UnetDown(256, 512)
        self.down5 = UnetDown(512, 512)
        self.down6 = UnetDown(512, 512)
        
        self.up1 = UnetUp(512, 512)
        self.up2 = UnetUp(1024, 512)
        self.up3 = UnetUp(1024, 256)
        self.up4 = UnetUp(512, 128)
        self.up5 = UnetUp(256, 64)
        
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        
        return self.final(u5)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, image_height, image_width = input_shape
        patch_h, patch_w = int(image_height / 2 ** 4), int(image_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        
        layers = []
        in_filters = channels
        for out_filters, stride, normalize, padding in \
            [(64, 2, False, 1), (128, 2, True, 1), (256, 2, True, 1), (256, 2, True, 1), (512, 1, True, 1)]:
            layers += self.discriminator_block(
                in_filters,
                out_filters,
                stride,
                padding,
                normalize
            )
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        
    
    def discriminator_block(self, in_filters, out_filters, stride, padding=1, normalize=True):
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride, padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        return self.model(x)
        

class DataSet(torchDataset):
    def __init__(self,
                 data_dir,
                 mean_data=None,
                 std_data=None,
                 img_size=(256, 256),
                 step="train") -> None:
        if not os.path.exists(data_dir):
            print("data dir - {} does not exist".format(data_dir))
        self.__input_dim = img_size  # height, width
        self._image_ids = []

        image_id = 1
        for orig_image_path in glob(os.path.join(data_dir, "origin") + "/*.jpg"):
            filename = orig_image_path.split("/")[-1]
            destroyed_filename = os.path.join(data_dir, "cut", filename)
            self._image_ids.append(
                [image_id, orig_image_path, destroyed_filename]
            )
        # print(self._image_ids)
        random.shuffle(self._image_ids)
        if mean_data is None or std_data is None:
            self._image_mean, self._image_std = self.cal_mean_std(self._image_ids)
        else:
            self._image_mean = mean_data
            self._image_std = std_data
        self._num_images = len(self._image_ids)
        
    def __len__(self):
        return len(self._image_ids)
        
    @property
    def input_dim(self):
        return self.__input_dim
    
    def get_random_samples(self, count_samples):
        if count_samples > self._num_images:
            count_samples = self._num_images
        select_imgs = self._image_ids[random.choice(self._num_images, count_samples, replace=False)]
        input_datas = []
        for item in select_imgs:
            destroyed_img_filename = item[2]
            img = cv2.imread(destroyed_img_filename)
            img = self.preproc(img, self.input_dim, self._image_mean, self._image_std)
            input_datas.append(img)
        
        return input_datas
    
    @classmethod
    def preproc(cls,
                img,
                input_dim,
                mean,
                std,
                swap=(2, 0, 1)):
        assert len(img.shape) == 3
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded_resized_image = cv2.resize(
            rgb_img, (input_dim[1], input_dim[0]), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        
        # normalzied
        for channel in range(0, 3):
            padded_resized_image[channel] = (padded_resized_image[channel] - mean[channel]) / std[channel]
        padded_resized_image = padded_resized_image.transpose(swap)
        
        return padded_resized_image
    
    def depreproc(self,
                  img,
                  swap=(1, 2, 0)):
        output_img = img.transpose(swap)
        for channel in range(0, 3):
            output_img[channel] = output_img[channel] * self._image_std[channel] + self._image_mean[channel]
        
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        return output_img
    
    def __getitem__(self, index):
        image_id, orig_image_path, destroyed_image_path = self._image_ids[index]
        # step 1. deal with image data
        label_img = cv2.imread(orig_image_path)
        input_img = cv2.imread(destroyed_image_path)
        
        label_img = self.preproc(
            label_img,
            self.__input_dim,
            self._image_mean,
            self._image_std)
        
        input_img = self.preproc(
            input_img,
            self.__input_dim,
            self._image_mean,
            self._image_std
        )
        
        return torch.tensor(input_img, dtype=torch.float32), \
               torch.tensor(label_img, dtype=torch.float32)
        
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
                interpolation=cv2.INTER_CUBIC)
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
                interpolation=cv2.INTER_CUBIC)
            # img = img.transpose((2, 0, 1))
            img_var = np.mean((img - mean_bgr) ** 2, axis=(0, 1))
            variances.append(img_var)

        std_bgr = np.sqrt(np.mean(variances, axis=0))
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
        std_rgb = np.array([std_bgr[2], std_bgr[1], std_bgr[0]])
        
        print("mean: {}".format(mean_rgb))
        print("std: {}".format(std_rgb))
        return mean_rgb, std_rgb


class MyTrainer:
    def __init__(self,
                 batch_size,
                 max_epoch,
                 data_dir,
                 basic_learning_rate,
                 model_save_dir,
                 use_l1=True,
                 warmup_epoches=5,
                 warmup_lr=0) -> None:
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.basic_learning_rate = basic_learning_rate
        self.warmup_lr = warmup_lr
        self.warmup_epoches = warmup_epoches
        self.epoch = 0
        self.model_name = "cargo_gan"
        self.use_l1 = use_l1
        self.sample_rate = 3
        self.sample_generate_dir = os.path.join(data_dir, "samples")
        if not os.path.exists(self.sample_generate_dir):
            os.makedirs(self.sample_generate_dir)
        
        train_data_dir = os.path.join(data_dir, "train")
        self.train_dataset = DataSet(
            train_data_dir,
            step="train"
        )
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # valid_data_dir = os.path.join(data_dir, "valid")
        # self.valid_dataset = DataSet(
        #     valid_data_dir,
        #     step="valid"
        # )
        # self.valid_data_loader = DataLoader(
        #     self.valid_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False
        # )
        
        input_shape = (3, self.train_dataset.input_dim[0], self.train_dataset.input_dim[1])
        self.generator_model = Generator(input_shape)
        self.generator_model.apply(self.weights_init)
        
        self.discriminator_model = Discriminator(input_shape)
        self.discriminator_model.apply(self.weights_init)
        
        self.is_cuda = torch.cuda.is_available()
        
        self.optimizer_G, self.optimizer_D = \
            self.get_optimizer(
                self.generator_model,
                self.discriminator_model,
                self.basic_learning_rate)
        # self.loss = self.loss_fn()
        
        input_size = self.train_dataset.input_dim
        print("input size: {}".format(input_size))
        print("generator model")
        summary(self.generator_model, (3, input_size[0], input_size[1]))
        
        print()
        print("discriminator model")
        summary(self.discriminator_model, (3, input_size[0], input_size[1]))
        
        self.model_save_dir = model_save_dir
        self.best_g_loss = 10000000000000000
        self.best_d_loss = 10000000000000000
        
    def weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)
            
    def loss_fn(self, model, inputs, labels, use_l1=True):
        # print(inputs.size(), labels.size())
        loss = nn.MSELoss()(inputs, labels)
        if use_l1:
            a = [parameter.view(-1) for parameter in model.parameters()]
            # print("a= {}".format(a))
            l1_lamda = 0.01
            l1_norm = torch.norm(
                torch.cat([parameter.view(-1) for parameter in model.parameters()]),
                p=1
            )
            loss += l1_lamda * l1_norm
        return loss

    def save_checkpoint(self, state, step, save_dir, model_name=""):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + step + "_best_ckpt.pth")
        torch.save(state, filename)
    
    def save_models(self, model, model_dir, optimizer, step, loss):
        ckpt_state = {
            "start_epoch": self.epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        }
        self.save_checkpoint(ckpt_state, step, model_dir, self.model_name)
    
    def get_optimizer(self, 
                      generator_model,
                      discriminator_model,
                      basic_learnig_rate):
        optimizer_G = torch.optim.Adam(
            generator_model.parameters(),
            lr=basic_learnig_rate,
            betas=(0.5, 0.999)
        )
        
        optimizer_D = torch.optim.Adam(
            discriminator_model.parameters(),
            lr=basic_learnig_rate,
            betas=(0.5, 0.999)
        )
        
        return optimizer_G, optimizer_D

    def train(self):
        if self.is_cuda:
            self.generator_model.cuda()
            self.discriminator_model.cuda()
            self.loss.cuda()
        
        self.generator_model.train()
        self.discriminator_model.train()
        for self.epoch in range(1, self.max_epoch + 1):
            self.itr = 0
            for self.itr, train_data in enumerate(self.train_data_loader):
                input_img = train_data[0]
                label_img = train_data[1]
                
                # Adversarial ground truths
                valid = torch.ones(input_img.size(0), *self.discriminator_model.output_shape)
                fake = torch.zeros(input_img.size(0), *self.discriminator_model.output_shape)
                
                if self.is_cuda:
                    input_img = input_img.cuda()
                    label_img = label_img.cuda()
                
                # -----------
                # train generator
                # -----------
                self.optimizer_G.zero_grad()
                gen_imgs = self.generator_model(input_img)
                
                # calculate loss for generator's ability
                g_loss = self.loss_fn(self.discriminator_model, self.discriminator_model(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()
                
                # -----------
                # train discriminator
                # -----------
                discrim_real_imgs = self.discriminator_model(label_img)
                discrim_gen_imgs = self.discriminator_model(gen_imgs.detach())
                real_loss = self.loss_fn(self.discriminator_model, discrim_real_imgs, valid)
                fake_loss = self.loss_fn(self.discriminator_model, discrim_gen_imgs, fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                
                d_loss.backward()
                self.optimizer_D.step()
                
                print("\rpoch:{}, itr:{}, g_loss: {:.04f}, d_loss: {:0.4f}".format(
                    self.epoch, self.itr, g_loss.item(), d_loss.item()), end=' ')
                
                if g_loss.item() < self.best_g_loss:
                    self.save_models(
                        self.generator_model,
                        self.model_save_dir,
                        self.optimizer_G,
                        "generator",
                        g_loss.item())
                    self.best_g_loss = g_loss.item()
                
                if d_loss.item() < self.best_d_loss:
                    self.save_models(
                        self.discriminator_model,
                        self.model_save_dir,
                        self.optimizer_D,
                        "discriminator",
                        d_loss.item())
                    self.best_d_loss = d_loss.item()
                    
            if self.epoch % self.sample_rate == 0:
                num_samples = 5
                input_data = self.train_dataset.get_random_samples(num_samples)
                input_data = np.array(input_data).reshape(num_samples, 3, self.train_dataset.input_dim[0], self.train_dataset.input_dim[1])
                input_data = torch.tensor(input_data)
                if self.is_cuda:
                    input_data.cuda()
                gen_imgs = self.generator_model(input_data).data
                for index in range(0, num_samples):
                    img = self.train_dataset.depreproc(gen_imgs[index])
                    if not os.path.exists(os.path.join(self.sample_generate_dir, f"{self.epoch}")):
                        os.makedirs(os.path.join(self.sample_generate_dir, f"{self.epoch}"))
                    cv2.imwrite(os.path.join(self.sample_generate_dir, f"{self.epoch}", f"{index}.png"), img)

def generate_train_data():
    images_dir = "/home/lutao/datasets/cargoes_data/labeled_data/train"
    labels_dir = "/home/lutao/datasets/cargoes_data/labeled_data/train"
    output_dir = "/home/lutao/datasets/cargoes_data/blocked_data/train"

    # preview_size = (960, 720)
    # object_labels_list = []
    
    for label_file in glob(labels_dir + "/*.xml"):
        try:
            content_tree = ET.parse(label_file)
            img_filename = content_tree.find("filename").text
            img_path = os.path.join(images_dir, img_filename)
            image = cv2.imread(img_path)
            image_show = deepcopy(image)
            for obj in content_tree.findall("object"):
                # id_code = obj.find("code").text
                # obj_name = obj.find("DeDescription").text
                # obj_level = obj.find("DefectLevel").text
                bbox = obj.find("bndbox")
                bbox_xyxy = [
                    math.floor(float(bbox.find("xmin").text)),
                    math.floor(float(bbox.find("ymin").text)),
                    math.floor(float(bbox.find("xmax").text)),
                    math.floor(float(bbox.find("ymax").text))
                ]
                # draw labels and bounding boxes
                image_show[bbox_xyxy[1]: bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 0
            
            output_img_filename_orign = os.path.join(output_dir, "origin", img_filename)
            output_img_filename_cut = os.path.join(output_dir, "cut", img_filename)
            cv2.imwrite(output_img_filename_orign, image)
            cv2.imwrite(output_img_filename_cut, image_show)
        except Exception as e:
            print(label_file, e)        


if __name__ == "__main__":
    # generate_train_data()
    trainer = MyTrainer(
        batch_size=2,
        max_epoch=40,
        data_dir="/home/lutao/datasets/cargoes_data/blocked_data",
        basic_learning_rate=0.0002,
        model_save_dir="./saved_models",
    )
    trainer.train()