from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 16

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_dir = f"{CURRENT_DIR}/pokemon-images-dataset-by-type/all"
    image_files = os.listdir(image_dir)

    image_features_list = []

    # --- 1) バッチごとに特徴ベクトルを取得し、リストに貯める ---
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        images = [Image.open(os.path.join(image_dir, f)) for f in batch_files]
        inputs = processor(images=images, return_tensors="pt")

        features = model.get_image_features(**inputs)  # shape: (batch_size, hidden_dim)
        
        image_features_list.append(features.detach().cpu())

        # --- 2) すべてのバッチを結合して NumPy 配列に変換 ---
        image_features = torch.cat(image_features_list, dim=0)  # shape: (N, hidden_dim)
        image_features = image_features.detach().numpy()

        # --- 3) PCAで次元圧縮 ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(image_features)  # shape: (N, 2)

        # --- 4) 散布図を描画 ---
        plt.figure(figsize=(6, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])

        # 任意で点にラベルをつける
        for i, (x, y) in enumerate(X_pca):
            plt.text(x, y, str(i), fontsize=8)

        plt.savefig(f"{CURRENT_DIR}/pca_result.png")
        plt.close()
