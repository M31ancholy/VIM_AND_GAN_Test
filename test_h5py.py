import h5py
with h5py.File("e14089s3_P53248.7.h5",'r')  as f:
    # 你可以遍历文件中的所有顶级组或数据集
    for key in f.keys():
        print(f[key].name)  # 打印组或数据集的名称 只有一个键
        print(f[key].shape)  # 如果是数据集，打印其形状
        # 如果需要，你可以读取数据集中的数据
        if isinstance(f[key], h5py.Dataset):
            data = f[key][:]
            print(data)