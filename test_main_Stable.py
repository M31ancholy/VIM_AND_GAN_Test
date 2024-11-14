import os
import sys
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
# 将 vim 目录添加到 PYTHONPATH
vim_path = os.path.join(project_root, 'vim')
sys.path.append(vim_path)
from vim import models_mamba

writer = SummaryWriter("Summary_Writer")
global step

import os
from collections import OrderedDict
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class NiiDataset(Dataset):
    def __init__(self, nii_dir, transform=None):
        """
        初始化 NiiDataset 类。

        参数:
        - nii_dir: 存储 .nii 或 .nii.gz 文件的目录路径。
        - transform: 图像变换函数，默认为 None。
        """
        self.nii_dir = nii_dir  # 设置 NIfTI 文件目录
        self.transform = transform  # 设置图像变换函数
        self.slices = []  # 存储文件路径和切片索引的列表

        # 遍历目录中的所有文件，仅存储文件路径和切片索引，而不加载图像数据
        for file_name in os.listdir(nii_dir):
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                nii_path = os.path.join(nii_dir, file_name)  # 获取 NIfTI 文件的完整路径
                nii_image = nib.load(nii_path)  # 加载 NIfTI 图像
                img_data = nii_image.shape  # 获取图像的形状信息

                # 记录文件路径及其每个切片的索引
                for i in range(img_data[2]):  # 假设图像是 3D 的
                    self.slices.append((file_name, nii_path, i))  # 将文件名、路径和切片索引添加到 slices 列表中

    def _load_slice(self, nii_path, file_name, index):
        """
        加载指定文件的特定切片

        参数:
        - nii_path: NIfTI 文件路径。
        - file_name: 文件名。
        - index: 切片索引。

        返回:
        - 3D 图像的特定切片。
        """
        nii_image = nib.load(nii_path)  # 加载 NIfTI 图像
        tmp = nii_image.get_fdata()  # 获取图像数据并缓存
        print(file_name+"  "+str(index))  # 打印加载的文件名
        return tmp[:, :, index]  # 返回指定切片

    def __len__(self):
        """
        返回数据集的总切片数。

        返回:
        - 切片的总数量。
        """
        return len(self.slices)

    def __getitem__(self, idx):
        """
        获取指定索引处的图像及其文件名。

        参数:
        - idx: 切片的索引。

        返回:
        - 处理后的 PIL 图像和文件名。
        """
        file_name, nii_path, slice_idx = self.slices[idx]  # 获取文件名、路径和切片索引
        slice_data = self._load_slice(nii_path, file_name, slice_idx)  # 加载切片数据

        # 对图像进行处理，这部分逻辑不应担心
        slice_data = np.rot90(slice_data)  # 旋转 90 度，以便图像方向正确

        # 处理最小值和最大值相等的情况
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if min_val != max_val:
            slice_data = (slice_data - min_val) / (max_val - min_val)  # 归一化
        else:
            slice_data = np.zeros_like(slice_data)  # 如果最小值和最大值相等，将所有值设置为 0

        slice_data = (slice_data * 255).astype(np.uint8)  # 转换为 8-bit 图像

        # 将数据转换为 PIL 图像以便保存为 PNG
        slice_image = Image.fromarray(slice_data)

        if self.transform:
            slice_image = self.transform(slice_image)  # 应用变换函数

        return slice_image, file_name  # 返回处理后的图像和文件名


def preprocess2(image): # 黑白1通道
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化
        transforms.Resize((224,224),antialias=True)
    ])
    image_tensor = transform(image).float()  # 确保数据类型为 float
    return image_tensor
def denormalize(image_tensor, mean, std):
    """
    反标准化图像张量
    :param image_tensor: 标准化后的图像张量
    :param mean: 标准化时使用的均值
    :param std: 标准化时使用的标准差
    :return: 反标准化后的图像张量
    """
    device = image_tensor.device  # 获取图像张量所在的设备
    mean = torch.tensor(mean).reshape(-1, 1, 1).to(device)
    std = torch.tensor(std).reshape(-1, 1, 1).to(device)
    denormalized_tensor = image_tensor * std + mean
    return denormalized_tensor

def convert_to_three_channels(tensor):
    """
    将形状为 (1, 224, 224) 的单通道张量转换为形状为 (3, 224, 224) 的三通道张量。

    参数:
    - tensor: 形状为 (1, 224, 224) 的 PyTorch 张量。

    返回:
    - 转换后的三通道张量，形状为 (3, 224, 224)。
    """
    # 检查输入张量的形状是否符合要求
    if tensor.shape != (1, 224, 224):
        raise ValueError("输入张量的形状必须是 (1, 224, 224)")

        # 使用 repeat 方法在通道维度上重复三次
    three_channel_tensor = tensor.repeat(1, 3, 1, 1)

    return three_channel_tensor
def extract_features(image_tensor, model): # 提取特征
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    return features

def train_gan(generator, discriminator, train_features, real_imgs, optimizer_G, optimizer_D, the_loss, device):
    global step

    batch_size = real_imgs.size(0)
    # valid = Variable(torch.ones(batch_size, 1).to(device), requires_grad=False)
    # fake = Variable(torch.zeros(batch_size, 1).to(device), requires_grad=False)

    valid = torch.full((batch_size, 1), 0.9).to(device)
    fake = torch.full((batch_size, 1), 0.0).to(device)  # 如果还需要一个全 0 的张量作为对比

    # 训练生成器
    optimizer_G.zero_grad()
    gen_imgs = generator(train_features)

    tmp = denormalize(gen_imgs.squeeze(0),mean=[0.5], std=[0.5])
    # print(gen_imgs.shape)
    # print(type(gen_imgs))
    writer.add_image('Generator', tmp, global_step=step, dataformats='CHW')




    g_loss = the_loss(discriminator(gen_imgs), valid) #bugged




    g_loss.backward()
    optimizer_G.step()

    # 训练判别器
    optimizer_D.zero_grad()
    real_loss = the_loss(discriminator(real_imgs), valid)
    fake_loss = the_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()



    return g_loss.item(), d_loss.item(), gen_imgs
def main():
    global step
    step = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 实例化VIM对象
    the_vision_mamba = models_mamba.vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False).to(device)
    # 加载预训练权重
    relative_path = "checkpoints/Vim-tiny-midclstok/vim_t_midclstok_ft_78p3acc.pth"
    pretrain_weights = torch.load(relative_path, map_location=device)
    the_vision_mamba.load_state_dict(pretrain_weights["model"], strict=True)
    the_vision_mamba.eval()

    # GAN 网络部分
    from DC_GAN import dcgan_modified
    generator = dcgan_modified.Generator().to(device)
    discriminator = dcgan_modified.Discriminator().to(device)
    the_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 加载 Nii 数据
    nii_dir = "/mnt/d/GIthub_Projects/MRI_Related/Datasets/CC-359/calgary-campinas_version-1.0.tar/calgary-campinas_version-1.0/calgary-campinas_version-1.0/CC359/Reconstructed/Original/Original"
    dataset = NiiDataset(nii_dir, transform=preprocess2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(20000):
        print("current epoch:{}".format(epoch))
        for idx, (slice_image, file_name) in enumerate(dataloader):
            # 这里 slice_image 是一个张量 (batch_size, channels, height, width)
            print(slice_image.shape)
            slice_image = slice_image[0].unsqueeze(0).to(device)
            slice_image = torch.nn.functional.interpolate(slice_image, size=(224, 224))
            slice_image = slice_image.squeeze(0)
            # print(slice_image.shape)
            slice_image_1ch = slice_image
            slice_image_3ch = convert_to_three_channels(slice_image)

            tmp = denormalize(slice_image_1ch, mean=[0.5], std=[0.5])
            writer.add_image("ori",tmp,step, dataformats='CHW')





            # 提取特征
            train_features = extract_features(slice_image_3ch, the_vision_mamba)
            # train_features = train_features.squeeze(0)
            print(train_features.shape)

            step += 1

            # 训练 GAN
            g_loss, d_loss, gen_imgs = train_gan(generator, discriminator, train_features, slice_image_1ch.unsqueeze(0), optimizer_G, optimizer_D, the_loss, device)
            print(f"Epoch [{epoch}/{9}] Batch [{idx}/{len(dataloader)}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
    writer.close()
if __name__ == "__main__":
    main()