import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from ccnb.bvse import bv_calculation
from ccnb import get_bvse


# 定义一个根据filename计算Ea值的函数（这里假设函数名为calculate_ea）
def calculate_ea(filename):
    # 在这里编写你的计算逻辑，并返回计算结果
    ea_value = filename + "_ea"  # 这里只是一个示例，你需要根据实际需求进行计算
    return ea_value


def batch_getEa(input_file, output_file):
    # 读取原始CSV文件
    input_file = "identify_group.csv"
    output_file = "your_output_file.csv"

    with open(input_file, "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # 在Ea列下方添加数据
    for row in data:
        filename = row["filename"]  # 获取当前行的filename值
        ea_value = calculate_ea(filename)  # 调用计算Ea值的函数
        row["Ea"] = ea_value  # 将计算得到的Ea值填入对应位置

    # 写入新的CSV文件
    fieldnames = data[0].keys()  # 获取字段名（表头）
    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头
        writer.writerows(data)  # 写入数据

    print("Ea值已成功计算并填入CSV文件中。")


def calculate_Ea_and_save_to_csv(folder_path, output_csv):
    data = []

    # 遍历文件夹中的 CIF 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".cif"):
            # 对每个 CIF 文件进行 Ea 计算
            ea_value = get_bvse(os.path.join(folder_path, file_name), 'Li', 1, 0.5)
            data.append({"File Name": file_name, "Ea Value": ea_value})

    # 将数据保存到 CSV 文件中
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def calculate_Ea(cif_file_path):
    # 在这里实现计算 Ea 的逻辑，假设返回一个随机值作为示例
    import random

    return random.randint(1, 100)


def extract_ea_elements_from_csv(csv_filename):
    x = []
    y = []
    z = []
    with open(csv_filename, "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)
        for row in data:
            ea_list = eval(row["Ea"])  # 将字符串转换为列表
            if len(ea_list) >= 2:  # 确保"Ea"列中有足够的元素
                x.append(ea_list[0])  # 将第一个元素添加到x列表中
                y.append(ea_list[1])  # 将第二个元素添加到y列表中
    for i in y:
        if i > 2:
            z.append(i)

    # 创建一个新的图形
    plt.figure(figsize=(10, 30), dpi=100)

    # 设置中文字体为SimHei
    plt.rcParams["font.family"] = "SimHei"

    # 绘制二维图，设置标题和坐标轴标签
    plt.scatter(x, y, c="red", s=100)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 3, 0.1))
    plt.title("二维图示例")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")

    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.show()
    return x, y


if __name__ == "__main__":
    # 使用示例
    folder_path = (
        'LiLaTaCl1000'
    )
    output_csv = "output1000.csv"
    calculate_Ea_and_save_to_csv(folder_path, output_csv)
