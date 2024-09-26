import os
import csv
from typing import Set, Optional
import pandas as pd

def generate_image_label_csv(
    root_dir: str,
    output_csv: str,
    include_full_path: bool = False,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    遍历指定的根目录，提取图片文件名、所属类别及对应的标签内容，并将结果保存为 CSV 文件。
    每个标签的类别和位置信息被拆分到单独的列中，每张图片占据一行。
    
    参数：
    - root_dir (str): 根目录路径，包含类别子文件夹。
    - output_csv (str): 输出的 CSV 文件路径。
    - include_full_path (bool): 是否在 CSV 中包含图片的完整路径。默认为 False。
    - encoding (str): CSV 文件的编码方式。默认为 'utf-8'。
    
    返回：
    - pd.DataFrame: 生成的 CSV 文件内容。
    """
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    data = []
    max_labels = 0

    # 遍历根目录下的每个类别文件夹，收集数据
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            images_dir = os.path.join(category_path, 'images')
            labels_dir = os.path.join(category_path, 'labels')

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                for image_file in os.listdir(images_dir):
                    image_path = os.path.join(images_dir, image_file)
                    if os.path.isfile(image_path):
                        _, ext = os.path.splitext(image_file)
                        if ext.lower() in image_extensions:
                            label_file = os.path.splitext(image_file)[0] + '.txt'
                            label_path = os.path.join(labels_dir, label_file)

                            if os.path.exists(label_path):
                                try:
                                    with open(label_path, 'r', encoding=encoding) as lf:
                                        label_content = lf.read().strip().replace('\n', ' ')
                                except Exception as e:
                                    print(f"Error reading label file {label_path}: {e}")
                                    label_content = ""
                            else:
                                print(f"Label file not found for image {image_file}")
                                label_content = ""

                            labels = []
                            if label_content:
                                parts = label_content.split()
                                # 每5个数字为一组
                                for i in range(0, len(parts), 5):
                                    if i + 4 < len(parts):
                                        cls = parts[i]
                                        x = parts[i+1]
                                        y = parts[i+2]
                                        w = parts[i+3]
                                        h = parts[i+4]
                                        labels.append((cls, x, y, w, h))
                            data.append({
                                'image_path': image_path if include_full_path else '',
                                'image_name': image_file,
                                'category': category,
                                'labels': labels
                            })
                            if len(labels) > max_labels:
                                max_labels = len(labels)

    print(f"最大标签数：{max_labels}")

    # 定义表头
    header = ['image_path', 'image_name', 'category'] if include_full_path else ['image_name', 'category']
    for i in range(1, max_labels + 1):
        header.extend([f'class_{i}', f'x_{i}', f'y_{i}', f'w_{i}', f'h_{i}'])

    # 写入 CSV 文件
    with open(output_csv, mode='w', newline='', encoding=encoding) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for entry in data:
            row = [
                entry['image_path'],
                entry['image_name'],
                entry['category']
            ] if include_full_path else [
                entry['image_name'],
                entry['category']
            ]
            for label in entry['labels']:
                row.extend(label)
            # 填充缺少的标签信息
            remaining = max_labels - len(entry['labels'])
            row.extend([''] * (remaining * 5))
            writer.writerow(row)

    print(f"CSV 文件已生成：{output_csv}")
    return pd.read_csv(output_csv)



def generate_image_cla_csv(
    root_dir: str,
    output_csv: str,
    include_full_path: bool = False,
    encoding: str = 'utf-8'
) -> None:
    """
    遍历指定的根目录，提取图片文件名、所属类别及对应的标签内容，并将结果保存为 CSV 文件。

    参数：
    - root_dir (str): 根目录路径，包含类别子文件夹。
    - output_csv (str): 输出的 CSV 文件路径。
    - image_extensions (Optional[Set[str]]): 支持的图片文件扩展名集合（如 {'.jpg', '.png'}）。默认为常见图片格式。
    - include_full_path (bool): 是否在 CSV 中包含图片的完整路径。默认为 False。
    - encoding (str): CSV 文件的编码方式。默认为 'utf-8'。

    返回：
    - None
    """

    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # 打开 CSV 文件以写入
    with open(output_csv, mode='w', newline='', encoding=encoding) as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头
        if include_full_path:
            writer.writerow(['image_path', 'image_name', 'category', 'label_content'])
        else:
            writer.writerow(['image_name', 'category', 'label_content'])

        # 遍历根目录下的每个类别文件夹
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                images_dir = os.path.join(category_path, 'images')
                labels_dir = os.path.join(category_path, 'labels')

                if os.path.exists(images_dir) and os.path.exists(labels_dir):
                    # 遍历 images 目录下的所有文件
                    for image_file in os.listdir(images_dir):
                        image_path = os.path.join(images_dir, image_file)
                        # 检查文件是否为图片
                        if os.path.isfile(image_path):
                            _, ext = os.path.splitext(image_file)
                            if ext.lower() in image_extensions:
                                # 构建对应的标签文件路径
                                label_file = os.path.splitext(image_file)[0] + '.txt'
                                label_path = os.path.join(labels_dir, label_file)

                                label_content = ''
                                if os.path.exists(label_path):
                                    try:
                                        with open(label_path, 'r', encoding=encoding) as lf:
                                            label_content = lf.read().strip().replace('\n', ' ')
                                    except Exception as e:
                                        label_content = f"Error reading label: {e}"
                                else:
                                    label_content = "Label file not found"

                                if include_full_path:
                                    writer.writerow([image_path, image_file, category, label_content])
                                else:
                                    writer.writerow([image_file, category, label_content])

    print(f"CSV 文件已生成：{output_csv}")




def check_multiple_labels(csv_file_path: str, output_filtered_csv: Optional[str] = None):
    """
    检查 CSV 文件中标签数量大于1的图片记录，并可选择将结果保存到新的 CSV 文件中。
    
    参数：
    - csv_file_path (str): 输入的 CSV 文件路径。
    - output_filtered_csv (Optional[str]): 输出筛选后 CSV 文件的路径。如果为 None，则不保存。
    
    返回：
    - pd.DataFrame: 包含标签数量大于1的图片记录。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    
    # 找到所有以 'class_' 开头的列
    class_columns = [col for col in df.columns if col.startswith('class_')]
    
    # 计算每行中 'class_' 列非空的数量，即标签数量
    df['label_count'] = df[class_columns].notnull().sum(axis=1)
    
    # 筛选标签数量大于1的行
    df_multiple_labels = df[df['label_count'] > 1]
    
    # 显示筛选结果
    print(f"总共有 {len(df_multiple_labels)} 张图片拥有超过1个标签。")
    print(df_multiple_labels.head())
    
    # 如果指定了输出路径，则保存筛选结果
    if output_filtered_csv:
        df_multiple_labels.to_csv(output_filtered_csv, index=False, encoding='utf-8')
        print(f"筛选后的 CSV 文件已保存到：{output_filtered_csv}")
    
    return df_multiple_labels

def main2():
    # 设置输入和输出的 CSV 文件路径
    input_csv = '/路径/到/您的/test_separate_columns.csv'  # 请根据实际路径修改
    output_csv = '/路径/到/您的/multiple_labels.csv'      # 请根据实际路径修改，或者设置为 None 不保存
    
    # 调用函数检查标签数量大于1的记录
    df_filtered = check_multiple_labels(input_csv, output_csv)
    
    # 如需进一步处理，可以在这里进行
    # 例如，查看特定列的数据
    # print(df_filtered[['image_name', 'label_count'] + class_columns])


    
def main():
    # 设置根目录路径
    root_directory = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/test_A/乳腺分类测试集A/A'  # 根据您的实际路径修改

    # 设置输出 CSV 文件路径
    output_csv_path = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/test_A/乳腺分类测试集A/A/test.csv'

    # 调用函数生成 CSV
    df = generate_image_label_csv(
        root_dir=root_directory,
        output_csv=output_csv_path,
        include_full_path=False,  # 如果需要包含完整路径，设为 True
        encoding='utf-8'
    )
    print(df.head())

if __name__ == "__main__":
    main()