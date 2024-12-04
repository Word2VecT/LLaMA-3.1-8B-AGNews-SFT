import json
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def load_data(jsonl_path, pkl_path):
    """加载 JSONL 和 PKL 文件数据"""
    labels = []
    predictions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data["label"])
            predictions.append(data["predict"])

    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)

    return labels, predictions, np.array(embeddings)


def plot_classification_tsne(embeddings, labels, predictions, unique_labels, save_path):
    """绘制 TSNE 效果图并标记预测错误的点"""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    correct_indices = []
    incorrect_indices = []

    # 分类点绘制
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        pred_indices = [i for i in indices if labels[i] == predictions[i]]
        incorrect_indices.extend([i for i in indices if labels[i] != predictions[i]])
        correct_indices.extend(pred_indices)
        plt.scatter(
            reduced_embeddings[pred_indices, 0],
            reduced_embeddings[pred_indices, 1],
            label=f"{label} (Correct)",
            alpha=0.7,
        )

    # 标记预测错误的点
    plt.scatter(
        reduced_embeddings[incorrect_indices, 0],
        reduced_embeddings[incorrect_indices, 1],
        color="red",
        label="Incorrect Predictions",
        marker="x",
    )

    plt.title("TSNE Visualization with Classification Results")
    plt.legend(loc="best")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def main(jsonl_path, pkl_path):
    # 加载数据
    labels, predictions, embeddings = load_data(jsonl_path, pkl_path)
    unique_labels = sorted(set(labels))

    # 绘制分类效果图
    plot_classification_tsne(
        embeddings, labels, predictions, unique_labels, save_path="classification_tsne_with_incorrect.png"
    )


jsonl_path = "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/generated_predictions.jsonl"  # 替换为实际 JSONL 文件路径
pkl_path = "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/test_embed.pkl"  # 替换为实际 PKL 文件路径
main(jsonl_path, pkl_path)
