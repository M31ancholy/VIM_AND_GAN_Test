from vim import models_mamba
import torch
import h5py
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
the_vision_mamba = models_mamba.VisionMamba( # 实例化Vim 对象
    patch_size=16,
    embed_dim=192,
    depth=24,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type='mean',
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="v2",
    if_cls_token=True,
    if_devide_out=True,
    use_double_cls_token=True,
    num_classes=0  # 设置 num_classes 为 0 以便仅提取特征
).to(device)
pretrain_weights = torch.load('checkpoints/Vim-tiny-midclstok/vim_t_midclstok_ft_78p3acc.pth',map_location=device)
the_vision_mamba.load_state_dict((pretrain_weights))
the_vision_mamba.eval()
def extract_features(image_tensor):
    with torch.no_grad():  # 禁用梯度计算
        features = the_vision_mamba.forward_features(image_tensor)
    return features

# 定义图像预处理函数
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    return transform(image)

# 从 h5 文件中加载图像数据
def load_image_from_h5(h5_file_path, dataset_name):
    with h5py.File(h5_file_path, 'r') as f:
        image_data = f[dataset_name][:]
    return image_data
h5_file_path = '/mnt/D/GIthub_Projects/MRI_Related/Datasets/CC-359/calgary-campinas_version-1.0.tar/calgary-campinas_version-1.0/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/train_val_12_channel/Train/e14089s3_P53248.7.h5'
train_dataset_name = 'train_dataset'
val_dataset_name = 'val_dataset'

train_image_data = load_image_from_h5(h5_file_path, train_dataset_name)
train_image_tensor = preprocess(train_image_data).unsqueeze(0)  # 添加 batch 维度
train_image_tensor = train_image_tensor.to(device)

# 加载并预处理验证图像
val_image_data = load_image_from_h5(h5_file_path, val_dataset_name)
val_image_tensor = preprocess(val_image_data).unsqueeze(0)  # 添加 batch 维度
val_image_tensor = val_image_tensor.to(device)

# 提取训练图像特征
train_features = extract_features(train_image_tensor)
print("Train features shape:", train_features.shape)  # 打印训练图像特征的形状

# 提取验证图像特征
val_features = extract_features(val_image_tensor)
print("Validation features shape:", val_features.shape)  # 打印验证图像特征的形状