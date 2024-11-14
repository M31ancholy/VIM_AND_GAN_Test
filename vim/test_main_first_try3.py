import os
import cv2
import torch
import h5py
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

from vim import models_mamba
from PIL import Image
from DC_GAN import dcgan_modified3
from torch.autograd import Variable
import argparse

#解析器
parser = argparse.ArgumentParser(description="VIM and GAN Integration")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

# 确定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化VIM对象
the_vision_mamba = models_mamba.vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False).to(device)
# 加载预训练权重
checkpoint_path = "/home/nextyear/Github_Projects/VIM_AND_GAN/checkpoints/Vim-tiny-midclstok/vim_t_midclstok_ft_78p3acc.pth"
pretrain_weights = torch.load(
    checkpoint_path,
    map_location=device)
the_vision_mamba.load_state_dict(pretrain_weights["model"], strict=True) #一个坑
the_vision_mamba.eval()
#模型部分设置完成

def extract_features(image_tensor): #提取特征
    with torch.no_grad():
        features = the_vision_mamba.forward_features(image_tensor)
    return features
# 定义图像预处理函数
def preprocess(image):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3)  # 标准化
    ])
    image_tensor = transform(image).float()  # 确保数据类型为 float
    return image_tensor

def preprocess2(image):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485] , std=[0.229])  # 标准化
    ])
    image_tensor = transform(image).float()  # 确保数据类型为 float
    return image_tensor
# 主函数
def main():
    test_png_path = "/mnt/d/GIthub_Projects/MRI_Related/Datasets/CC-359/calgary-campinas_version-1.0.tar/calgary-campinas_version-1.0/calgary-campinas_version-1.0/CC359/Reconstructed/Original/Original/CC0001_philips_15_55_M.nii/CC0001_slices/CC0001_philips_15_55_M_slice_000.png"
    test_image = Image.open(test_png_path)
    test_image = test_image.resize((224,224))
#黑白的现在就搞出来
    test_image_black_and_white = preprocess2(test_image)

    input_tensor2 = test_image_black_and_white
    input_tensor2 = input_tensor2.squeeze(0)

    # 将 Tensor 转换为 PIL 图像
    to_pil = ToPILImage()
    image = to_pil(input_tensor2)

    # 保存图像
    image.save('output_image_ori.png')

    print(test_image_black_and_white.shape)


    test_image_array = np.array(test_image)
    test_image_array = np.stack([test_image_array]*3,axis=-1)
    test_image_tensor = preprocess(test_image_array)
    test_image_tensor = test_image_tensor.unsqueeze(0).float().to(device)
    train_features = extract_features(test_image_tensor)

    print("Train features shape:", train_features.shape)
    print(type(train_features))

#GAN 网络部分
    # hyper parameters
    channels = 1
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999









    generator = dcgan_modified3.Generator().to(device)
    discriminator = dcgan_modified3.Discriminator().to(device)
    #初始化损失函数和生成器和判别器
    the_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = lr,betas=(b1,b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    batch_size = test_image_black_and_white.size(0)
    valid = Variable(torch.ones(batch_size, 1).to(device), requires_grad=False)
    fake = Variable(torch.zeros(batch_size, 1).to(device), requires_grad=False)
    test_image_black_and_white = test_image_black_and_white.unsqueeze(1)
    real_imgs = Variable(test_image_black_and_white.to(device))


    for i in range(1000):
        print(i)
        # # 训练生成器
        optimizer_G.zero_grad()
        gen_imgs = generator(train_features)
        print(gen_imgs.shape)  # 应输出torch.Size([1, 3, 224, 224])

        input_tensor = gen_imgs
        input_tensor = input_tensor.squeeze(0)

        # 将 Tensor 转换为 PIL 图像
        to_pil = ToPILImage()
        image = to_pil(input_tensor)

        # 保存图像
        image.save('output_image.png')

        g_loss = the_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()
        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = the_loss(discriminator(real_imgs), valid)
        fake_loss = the_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

if __name__ == "__main__":
    main()
