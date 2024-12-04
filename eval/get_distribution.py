import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def load_data(jsonl_path, pkl_path):
    """加载 JSONL 和 PKL 文件数据"""
    labels = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data["label"])

    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)

    return labels, np.array(embeddings)


def plot_tsne(embeddings, labels, unique_labels, title, save_path):
    """绘制 TSNE 效果图并保存"""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label, alpha=0.7)

    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid()
    plt.savefig(save_path)


def main(jsonl_path, pkl_path):
    # 加载数据
    labels, embeddings = load_data(jsonl_path, pkl_path)
    unique_labels = sorted(set(labels))

    # 绘制分类效果图
    plot_tsne(
        embeddings,
        labels,
        unique_labels,
        title="TSNE Visualization of Classification Results",
        save_path="classification_tsne.png",
    )

    # 绘制标签分布图
    plot_tsne(embeddings, labels, unique_labels, title="TSNE Visualization by Labels", save_path="label_tsne.png")


# 示例调用
jsonl_path = "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/generated_predictions.jsonl"  # 替换为实际 JSONL 文件路径
pkl_path = "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/test_embed.pkl"  # 替换为实际 PKL 文件路径
main(jsonl_path, pkl_path)
