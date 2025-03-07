# -*- coding: utf-8 -*-
"""CLIP_pokemon_dataset_

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cLgAUJ8PLAG1fdnKab_tSUbRH4UTrRid
"""

from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
CURRENT_DIR = ''
BATCH_SIZE = 16

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_dir = f"{CURRENT_DIR}pokemon-images-dataset-by-type/all"
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
            plt.text(x, y, image_files[i], fontsize=8)

        plt.savefig(f"{CURRENT_DIR}pca_result_{i}.png")
        plt.close()

base_dir = f"{CURRENT_DIR}pokemon-images-dataset-by-type/"
type_candidates = os.listdir(base_dir)
types = []
for candidate in type_candidates:
  if candidate != "all" and candidate != "README.md":
    types.append(candidate)
# print(types)
poke_dict = {}
for poke_type in types:
  poke_dict[poke_type] = set(os.listdir(f'{base_dir}{poke_type}/'))

def find_pokemon_type(pokemon_name: str) -> int:
  """
  find the type of pokemon
  """
  for poke_type in types:
    if pokemon_name in poke_dict[poke_type]:
      return poke_type

# --- タイプごとの色を定義 ---
TYPE_COLORS = {
    "fire": "red",
    "water": "blue",
    "grass": "green",
    "electric": "yellow",
    "psychic": "purple",
    "ice": "cyan",
    "dragon": "orange",
    "dark": "black",
    "fairy": "pink",
    "fighting": "brown",
    "flying": "skyblue",
    "ghost": "gray",
    "ground": "sandybrown",
    "poison": "violet",
    "rock": "gold",
    "bug": "lime",
    "steel": "silver",
    "normal": "tan",
}

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def display_pokemon_by_radius(file_name, radius, image_dir, features, names, k=10):
    """
    指定されたファイル名を基点として、
    半径r内と半径r外のポケモン画像を表示する。

    Parameters:
    - file_name: 基準となるファイル名。
    - radius: 半径r。
    - image_dir: 画像ディレクトリへのパス。
    - features: 特徴ベクトル (Numpy 配列)。
    - names: ファイル名リスト。
    - k: 表示する画像の数 (デフォルトは10)。
    """
    try:
        # 基準となるファイルのインデックスを取得
        base_idx = names.index(file_name)
        base_feature = features[base_idx]
    except ValueError:
        print(f"Error: {file_name} is not found in the dataset.")
        return

    # 各特徴ベクトルとのユークリッド距離を計算
    distances = np.linalg.norm(features - base_feature, axis=1)

    # 半径r以内のインデックスを取得し、距離でソート
    within_radius_idx = np.where(distances <= radius)[0]
    within_radius_idx = within_radius_idx[np.argsort(distances[within_radius_idx])]

    # 半径r外のインデックスを取得し、距離でソート
    outside_radius_idx = np.where(distances > radius)[0]
    outside_radius_idx = outside_radius_idx[np.argsort(distances[outside_radius_idx])]
    outside_radius_idx = outside_radius_idx[::-1]
    # 半径r以内のポケモンを表示
    print(f"Displaying {len(within_radius_idx[:k])} Pokémon within radius {radius}:")
    fig, ax = plt.subplots(1, min(k, len(within_radius_idx)), figsize=(15, 5))
    for i, idx in enumerate(within_radius_idx[:k]):
        image_path = os.path.join(image_dir, names[idx])
        image = Image.open(image_path)
        ax[i].imshow(image)
        ax[i].axis("off")
        ax[i].set_title(f"{names[idx]}\nDist: {distances[idx]:.2f}")
    plt.savefig(f"{file_name}_closer_{radius}.png")
    plt.close()

    # 半径r外のポケモンを表示
    print(f"Displaying {len(outside_radius_idx[:k])} Pokémon outside radius {radius}:")
    fig, ax = plt.subplots(1, min(k, len(outside_radius_idx)), figsize=(15, 5))
    for i, idx in enumerate(outside_radius_idx[:k]):
        image_path = os.path.join(image_dir, names[idx])
        image = Image.open(image_path)
        ax[i].imshow(image)
        ax[i].axis("off")
        ax[i].set_title(f"{names[idx]}\nDist: {distances[idx]:.2f}")
    plt.savefig(f"{file_name}_greater_{radius}.png")
    plt.close()

# 使用例
# display_pokemon_by_radius("example.png", 10.0, "path/to/images", X_pca, names, k=10)

from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
CURRENT_DIR = ''
BATCH_SIZE = 16

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_dir = f"{CURRENT_DIR}pokemon-images-dataset-by-type/all"
    image_files = os.listdir(image_dir)

    image_features_list = []
    labels = []

    # --- 1) バッチごとに特徴ベクトルを取得し、リストに貯める ---
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        images = [Image.open(os.path.join(image_dir, f)) for f in batch_files]
        inputs = processor(images=images, return_tensors="pt")

        features = model.get_image_features(**inputs)  # shape: (batch_size, hidden_dim)
        image_features_list.append(features.detach().cpu())

        # ファイル名からタイプラベルを取得
        labels.extend([find_pokemon_type(f) for f in batch_files])

    # --- 2) すべてのバッチを結合して NumPy 配列に変換 ---
    image_features = torch.cat(image_features_list, dim=0)  # shape: (N, hidden_dim)
    image_features = image_features.detach().numpy()

    # --- 3) PCAで次元圧縮 ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(image_features)  # shape: (N, 2)

    # --- 4) 散布図を描画 ---
    plt.figure(figsize=(10, 10))

    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, color=TYPE_COLORS.get(label, "gray"))

    # ラベルと凡例
    plt.legend(loc="best")
    plt.title("PCA Visualization of Pokémon Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()

    # 結果を保存
    plt.savefig(f"{CURRENT_DIR}pca_result_by_type.png")
    plt.close()

from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用
from tqdm import tqdm

# --- タイプごとの色を定義 ---
TYPE_COLORS = {
    "fire": "red",
    "water": "blue",
    "grass": "green",
    "electric": "yellow",
    "psychic": "purple",
    "ice": "cyan",
    "dragon": "orange",
    "dark": "black",
    "fairy": "pink",
    "fighting": "brown",
    "flying": "skyblue",
    "ghost": "gray",
    "ground": "sandybrown",
    "poison": "violet",
    "rock": "gold",
    "bug": "lime",
    "steel": "silver",
    "normal": "tan",
}

CURRENT_DIR = ''
BATCH_SIZE = 16

if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_dir = f"{CURRENT_DIR}pokemon-images-dataset-by-type/all"
    image_files = os.listdir(image_dir)

    image_features_list = []
    labels = []
    names = []
    # --- 1) バッチごとに特徴ベクトルを取得し、リストに貯める ---
    for i in tqdm(range(0, len(image_files), BATCH_SIZE)):
        batch_files = image_files[i : i + BATCH_SIZE]
        images = [Image.open(os.path.join(image_dir, f)).convert("RGBA") for f in batch_files]
        inputs = processor(images=images, return_tensors="pt")

        features = model.get_image_features(**inputs)  # shape: (batch_size, hidden_dim)
        image_features_list.append(features.detach().cpu())

        # ファイル名からタイプラベルを取得
        labels.extend([find_pokemon_type(f) for f in batch_files])
        names.extend([f for f in batch_files])

    # --- 2) すべてのバッチを結合して NumPy 配列に変換 ---
    image_features = torch.cat(image_features_list, dim=0)  # shape: (N, hidden_dim)
    image_features = image_features.detach().numpy()

    # --- 3) PCAで次元圧縮 ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(image_features)  # shape: (N, 3)

    # --- 4) 3D散布図を描画 ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  # 3Dプロット設定
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2],
            label=label, color=TYPE_COLORS.get(label, "gray"), alpha=0.8
        )
        # 各点に名前ラベルを付ける
        for i in idx:
            ax.text(
                X_pca[i, 0], X_pca[i, 1], X_pca[i, 2],
                names[i], fontsize=8
            )


    # ラベルと凡例
    ax.set_title("3D PCA Visualization")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.legend()

    # 結果を保存
    plt.savefig(f"{CURRENT_DIR}pca_result_3d_by_type_with_labels.png")
    plt.close()



display_pokemon_by_radius("pikachu.png", 1, f"{CURRENT_DIR}pokemon-images-dataset-by-type/all/", X_pca, names, k=10)

from sklearn.cluster import KMeans

# --- 5) K-Means クラスタリング ---
n_clusters = 10 # ポケモンのタイプの数に基づいてクラスタ数を決定
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)  # shape: (N,)

# --- 6) クラスタごとの分布を 3D 散布図にプロット ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')  # 3D プロット設定

# クラスタごとに色分けしてプロット
for cluster in range(n_clusters):
    idx = [i for i, c in enumerate(clusters) if c == cluster]
    ax.scatter(
        X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2],
        label=f"Cluster {cluster}", alpha=0.8
    )

# クラスタ中心をプロット
centers = kmeans.cluster_centers_  # shape: (n_clusters, 3)
ax.scatter(
    centers[:, 0], centers[:, 1], centers[:, 2],
    s=300, c="red", marker="X", label="Cluster Centers"
)

# ラベルと凡例
ax.set_title("3D PCA Visualization with K-Means Clustering")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.legend()

# 結果を保存
plt.savefig(f"{CURRENT_DIR}kmeans_pca_result_3d_{n_clusters}.png")
plt.close()

# --- 7) クラスタリング結果の割り当てを出力 ---
cluster_results = list(zip(names, clusters, labels))
for name, cluster, label in cluster_results[:10]:  # 最初の10件だけ表示
    print(f"Image: {name}, Cluster: {cluster}, True Label: {label}")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def display_pokemon_per_cluster(cluster_results, n_clusters, max_per_cluster=10, image_dir=None):
    """
    各クラスターに属するポケモンを最大 max_per_cluster 個、グリッド上に画像を表示する関数。

    Parameters:
        cluster_results (list of tuples): [(file_name, cluster, true_label), ...] の形式のリスト。
        n_clusters (int): クラスタ数。
        max_per_cluster (int): 各クラスタに表示するポケモンの最大数。
        image_dir (str): 画像ディレクトリへのパス。
    """
    cluster_dict = {i: [] for i in range(n_clusters)}

    # クラスタごとにポケモンを分類
    for name, cluster, label in cluster_results:
        cluster_dict[cluster].append((name, label))

    # 各クラスタのポケモンをグリッド表示
    for cluster, items in cluster_dict.items():
        print(f"\nCluster {cluster}:")
        items_to_display = items[:max_per_cluster]
        fig, axes = plt.subplots(1, len(items_to_display), figsize=(15, 5))
        if len(items_to_display) == 1:
            axes = [axes]  # 1つの場合でもリスト化して処理を統一

        for ax, (name, label) in zip(axes, items_to_display):
            if image_dir:
                image_path = os.path.join(image_dir, name)
                try:
                    image = Image.open(image_path)
                    ax.imshow(image)
                except FileNotFoundError:
                    print(f"Error: File {name} not found in {image_dir}.")
                    ax.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center')
            ax.axis("off")
            ax.set_title(f"{name}\nType: {label}")

        plt.suptitle(f"Cluster {cluster}", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"")

# クラスタリング結果を表示
# cluster_results: [(file_name, cluster, true_label), ...] 形式のリスト
# n_clusters: K-Meansで設定したクラスタ数

display_pokemon_per_cluster(cluster_results, n_clusters, max_per_cluster=10, image_dir=f"{CURRENT_DIR}pokemon-images-dataset-by-type/all/")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# --- タイプごとの色を定義 (例) ---
TYPE_COLORS = {
    "fire": "red",
    "water": "blue",
    "grass": "green",
    "electric": "yellow",
    "psychic": "purple",
    "ice": "cyan",
    "dragon": "orange",
    "dark": "black",
    "fairy": "pink",
    "fighting": "brown",
    "flying": "skyblue",
    "ghost": "gray",
    "ground": "sandybrown",
    "poison": "violet",
    "rock": "gold",
    "bug": "lime",
    "steel": "silver",
    "normal": "tan",
}

CURRENT_DIR = ""

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(labels)  # 0 ~ (クラス数-1)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    image_features, y_all, names, test_size=0.2, random_state=42
)

# numpy -> torch tensor
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).long()
X_test_torch = torch.from_numpy(X_test).float()
y_test_torch = torch.from_numpy(y_test).long()

class PokemonTypeClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=18):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

    def extract_features(self, x):
        """
        最終層直前( layer4 の出力 )の特徴を返す
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

model = PokemonTypeClassifier(input_dim=512, hidden_dim=256, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
epochs = 50
batch_size = 32

for epoch in range(epochs):
    # 学習用データをランダムにシャッフル
    permutation = torch.randperm(X_train_torch.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_torch.size(0), batch_size):
        indices = permutation[i : i + batch_size]
        batch_x = X_train_torch[indices]
        batch_y = y_train_torch[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

model.eval()
with torch.no_grad():
    outputs_test = model(X_test_torch)
    predictions = torch.argmax(outputs_test, dim=1)
    accuracy = (predictions == y_test_torch).float().mean().item()
    print(f"Test Accuracy: {accuracy*100:.2f}%")

with torch.no_grad():
    # (a) 学習データの最終層直前の特徴
    features_train = model.extract_features(X_train_torch)  # shape: (N_train, hidden_dim)
    # (b) テストデータの最終層直前の特徴
    features_test = model.extract_features(X_test_torch)    # shape: (N_test, hidden_dim)

# PCA の学習は学習データ上で行い、テストデータは同じ変換を適用
pca = PCA(n_components=3)
features_train_np = features_train.numpy()
features_test_np  = features_test.numpy()

pca.fit(features_train_np)
train_pca_3d = pca.transform(features_train_np)
test_pca_3d  = pca.transform(features_test_np)

# -------------------------------------------------------------------------------------
#    プロット（3D scatter）
#    同じ図上に、学習データ(●)とテストデータ(▲)を描画し、色はタイプによって分ける
# -------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# ラベルを文字列に戻す (可視化用)
train_labels_str = label_encoder.inverse_transform(y_train)
test_labels_str  = label_encoder.inverse_transform(y_test)
train_labels_set = set(train_labels_str)
train_labels_set.remove(None)
test_labels_set = set(test_labels_str)
test_labels_set.remove(None)
unique_labels = sorted(train_labels_set| test_labels_set)

for label in unique_labels:
    train_idx = [i for i, l in enumerate(train_labels_str) if l == label]
    ax.scatter(
        train_pca_3d[train_idx, 0],
        train_pca_3d[train_idx, 1],
        train_pca_3d[train_idx, 2],
        label=label + " (train)",
        color=TYPE_COLORS.get(label, "gray"),
        alpha=0.8,
        marker='o',  # 学習データは丸
    )
    test_idx = [i for i, l in enumerate(test_labels_str) if l == label]
    ax.scatter(
        test_pca_3d[test_idx, 0],
        test_pca_3d[test_idx, 1],
        test_pca_3d[test_idx, 2],
        color=TYPE_COLORS.get(label, "gray"),
        alpha=0.8,
        marker='^',  # テストデータは三角
    )

ax.set_title("3D PCA Visualization (Features Before Last Layer)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}pca_result_3d_final_features_train_test.png")
plt.show()
plt.close()

