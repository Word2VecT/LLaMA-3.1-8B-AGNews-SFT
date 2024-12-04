import json
from collections import defaultdict


def calculate_metrics(file_path, labels):
    total = 0
    correct = 0
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)
    total_per_class = defaultdict(int)

    # 读取 JSONL 文件
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 加载每一行的 JSON 对象
            data = json.loads(line.strip())
            total += 1
            label = data["label"]
            predict = data["predict"]
            total_per_class[label] += 1

            if label in predict:
                correct += 1
                true_positive[label] += 1
            else:
                false_positive[predict] += 1
                false_negative[label] += 1

    # 计算分类指标
    accuracy = correct / total if total > 0 else 0
    precision, recall, f1, per_class_accuracy = {}, {}, {}, {}

    for label in labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        total_label = total_per_class[label]

        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[label] = (
            (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
            if (precision[label] + recall[label]) > 0
            else 0
        )
        per_class_accuracy[label] = tp / total_label if total_label > 0 else 0

    # 计算整体宏观指标
    macro_precision = sum(precision.values()) / len(labels)
    macro_recall = sum(recall.values()) / len(labels)
    macro_f1 = sum(f1.values()) / len(labels)

    # 输出结果
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print("\nMetrics per class:")
    for label in labels:
        print(f"Label: {label}")
        print(f"  Accuracy: {per_class_accuracy[label]:.2%}")
        print(f"  Precision: {precision[label]:.2%}")
        print(f"  Recall: {recall[label]:.2%}")
        print(f"  F1 Score: {f1[label]:.2%}")
    print("\nOverall Macro Metrics:")
    print(f"  Macro Precision: {macro_precision:.2%}")
    print(f"  Macro Recall: {macro_recall:.2%}")
    print(f"  Macro F1 Score: {macro_f1:.2%}")

    return accuracy, precision, recall, f1, macro_precision, macro_recall, macro_f1


# 定义四分类的标签
labels = ["1: World", "2: Sports", "3: Business", "4: Sci/Tech"]

# 示例调用
file_path = (
    "/mnt/petrelfs/tangzinan/LLaMA-Factory/news_train/llama/final/generated_predictions.jsonl"  # 替换为实际文件路径
)
calculate_metrics(file_path, labels)
