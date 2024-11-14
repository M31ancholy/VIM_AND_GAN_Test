import os
import cv2
import torch
import h5py
import numpy as np
import torchvision.transforms as transforms
from vim import models_mamba
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
        transforms.ToTensor(),  # 转换为张量
        # transforms.Resize((224, 224)),  # 调整图像大小
        transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3)  # 标准化
    ])
    image_tensor = transform(image).float()  # 确保数据类型为 float
    return image_tensor


# 从 h5 文件中加载图像数据
def load_image_from_h5(h5_file_path, dataset_name):
    with h5py.File(h5_file_path, 'r') as f:
        image_data = f[dataset_name][:]
    return image_data


# 从k空间数据中提取图像域数据
def kspace_to_image(kspace_data):
    image_data = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_data)))  # 应用傅里叶变换
    image_data = np.abs(image_data)  # 获取幅度信息
    return image_data


# 将单通道灰度图像转换为三通道RGB图像
def convert_to_rgb(image_data):
    rgb_image = np.stack([image_data] * 3, axis=-1)  # 将单通道图像复制到三通道
    return rgb_image


# 主函数
def main():
    h5_file_path = '/mnt/d/GIthub_Projects/MRI_Related/Datasets/CC-359/calgary-campinas_version-1.0.tar/calgary-campinas_version-1.0/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/train_val_12_channel/Train/e14089s3_P53248.7.h5' #可以修改一下
    train_dataset_name = 'kspace' #CC359数据集的
    #调用函数 加载h5数据
    train_image_data = load_image_from_h5(h5_file_path, train_dataset_name)

    slice_index = 0  # 选择第0个切片
    slice_data = train_image_data[:, :, slice_index, :]  # 选择特定切片的k空间数据
    slice_data = slice_data[:, :, 0]  # 选择第一个通道进行傅里叶变换
    image_data = kspace_to_image(slice_data)  # k空间转换为图像域数据
    print("Image data shape after FFT:", image_data.shape)

    # 将单通道灰度图像转换为三通道RGB图像
    rgb_image = convert_to_rgb(image_data)
    print("RGB image shape:", rgb_image.shape)

    rgb_image_resized = cv2.resize(rgb_image, (224, 224))  # 调整图像大小
    print("Resized RGB image shape:", rgb_image_resized.shape)

    train_image_tensor = preprocess(rgb_image_resized)  # 预处理图像

    # 添加 batch 维度并确保类型为 float
    train_image_tensor = train_image_tensor.unsqueeze(0).float().to(device)

    # 提取训练图像特征
    train_features = extract_features(train_image_tensor)
    print("Train features shape:", train_features.shape)


if __name__ == "__main__":
    main()
