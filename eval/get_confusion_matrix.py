import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(file_path, labels):
    y_true = []
    y_pred = []

    # 读取 JSONL 文件并提取真实标签和预测标签
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            y_true.append(data["label"])
            y_pred.append(data["predict"])

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 使用 sklearn 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.savefig("/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/matrix.png")


# 定义四分类的标签
labels = ["1: World", "2: Sports", "3: Business", "4: Sci/Tech"]

# 示例调用
file_path = (
    "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/generated_predictions.jsonl"  # 替换为实际文件路径
)
plot_confusion_matrix(file_path, labels)
