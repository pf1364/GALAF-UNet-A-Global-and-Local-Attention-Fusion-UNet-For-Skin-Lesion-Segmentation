import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def compute_isic_mean_std(img_dir: str):
    """
    Args:
        img_dir: 存放训练图像的文件夹路径。
    """
    img_channels = 3

    # 1. 检查路径是否存在
    assert os.path.exists(img_dir), f"Image directory: '{img_dir}' does not exist."

    # 2. 获取所有.jpg图像的文件名列表
    # 使用 .lower() 来确保能找到 .JPG, .Jpg 等文件
    img_name_list = [i for i in os.listdir(img_dir) if i.lower().endswith(".jpg")]

    if not img_name_list:
        print(f"Warning: No .jpg images found in '{img_dir}'.")
        return

    print(f"Found {len(img_name_list)} images to process.")

    # 3. 初始化累加器
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    # 4. 遍历图像并累加均值和标准差
    for img_name in tqdm(img_name_list, desc="Calculating Mean/Std"):
        img_path = os.path.join(img_dir, img_name)

        # 打开图像，转换为RGB，并归一化到[0, 1]
        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        # 计算单张图片的均值和标准差，并在通道维度上累加
        # axis=(0, 1) 表示在高度和宽度维度上计算
        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    # 5. 计算最终的平均均值和平均标准差
    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)

    # 打印结果，格式化输出以便复制粘贴
    print("\n" + "=" * 40)
    print("Calculation Complete!")
    print(f"Mean (R, G, B): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std (R, G, B):  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("\nFormatted for direct use in code:")
    print(f"mean = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"std = ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    print("=" * 40)


if __name__ == '__main__':
    # ---路径 ---
    isic_img_directory = "E:/ayan/segmentation/dataset/ISIC/ISIC2017_split/train/images"
    compute_isic_mean_std(isic_img_directory)
