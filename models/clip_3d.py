from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 16

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_dir = f"{CURRENT_DIR}/pokemon-images-dataset-by-type/all"
    image_files = os.listdir(image_dir)
    image_features_list = []
    
    for i in range(0, len(image_files), BATCH_SIZE):
        images = [Image.open(os.path.join(image_dir, image_files[j])).convert("RGB") \
                  for j in range(i, min(i + BATCH_SIZE, len(image_files)))]
        inputs = processor(images=images, return_tensors="pt")
        
        with torch.no_grad():  # 勾配追跡を無効化
            features = model.get_image_features(**inputs).cpu()
        image_features_list.append(features)

    # 特徴量を結合
    image_features = torch.cat(image_features_list, dim=0)

    # NumPy 配列に変換
    image_features = image_features.detach().numpy()

    # PCA
    pca = PCA(n_components=3)  # 3次元に設定
    pca_result = pca.fit_transform(image_features)

    # 3Dプロット
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # 3Dプロット設定

    for idx in range(len(image_files)):
        x, y, z = pca_result[idx]
        ax.scatter(x, y, z, label=image_files[idx])  # 散布図
        ax.text(x, y, z, image_files[idx], fontsize=8)  # ラベル

    ax.set_title("3D PCA Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.legend()
    plt.savefig(f"{CURRENT_DIR}/pca_3d_result.png")
    plt.show()
