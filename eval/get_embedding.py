import argparse
import json  # 新增导入 JSON 模块
import os
import pickle

import numpy as np
from datadreamer import DataDreamer
from datadreamer.embedders import OpenAIEmbedder


# 指定并读入命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="输入的 JSON 文件路径")
parser.add_argument("--output_file", type=str, required=True, help="输出的 embedding 文件路径")
parser.add_argument("--idx", type=int, default=0, help="输出文件夹的索引")
args = parser.parse_args()

# 设置对应 API
Baseurl = "https://api.claudeshop.top/v1"
Skey = "sk-o3TBeafjjRJxvBaLS4Ca8UN6aFIIsv0FkrRfIDces4hYBhGD"
os.environ["OPENAI_API_KEY"] = Skey
os.environ["OPENAI_BASE_URL"] = Baseurl

# 合法性检查
if args.input_file is None:
    print("请指定 input_file")
    exit(1)
if args.output_file is None:
    print("请指定 output_file")
    exit(1)

# 载入数据
with open(args.input_file, "r", encoding="utf-8") as f:
    try:
        json_data = json.load(f)
        # 假设 JSON 文件是一个列表，每个元素是一个字典
        data = [item["input"] for item in json_data]
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e}")
        exit(1)
    except Exception as e:
        print(f"读取 JSON 文件时出错: {e}")
        exit(1)

if not data:
    print("未找到任何数据。")
    exit(1)

# 如果数据过长会报错，所以可以加上这里的截断，如果不长就不需要
# for i in range(len(data)):
#     if len(data[i])>=7000:
#         data[i]=data[i][:7000]
#         print("length error!")

# 确保输出目录存在
output_dir = f"./output/{args.idx}th/"
os.makedirs(output_dir, exist_ok=True)

# 开始 embedding
with DataDreamer(output_dir):
    # 载入 embedder
    Embedder = OpenAIEmbedder(
        model_name="text-embedding-ada-002",  # 确认模型名称是否正确
    )
    # embed
    embedding = Embedder.run(texts=data, truncate=True)
    embedding = np.array(embedding)

    print(f"完成！\nembedding.shape 是: {embedding.shape}")

    with open(args.output_file, "wb") as f_out:
        pickle.dump(embedding, f_out)
