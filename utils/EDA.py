import os
import pandas as pd
import plotly.express as px

def visualize_category_distribution(main_dir='./'):
    # 获取所有类别文件夹
    categories = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    category_counts = []

    total_files = 0  # 计算总文件数
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    for category in categories:
        category_path = os.path.join(main_dir, category)
        images_path = os.path.join(category_path, 'images')

        if not os.path.exists(images_path):
            print(f"警告: 类别 '{category}' 中缺少 'images' 目录。")
            count = 0
        else:
            # 列出 images/ 目录中的所有文件，并过滤图像文件
            files = [
                f for f in os.listdir(images_path)
                if os.path.isfile(os.path.join(images_path, f)) 
                and not f.startswith('.') 
                and os.path.splitext(f)[1].lower() in image_extensions
            ]
            count = len(files)

        total_files += count
        category_counts.append({'类别': category, '文件数量': count})

    df = pd.DataFrame(category_counts)
    
    # 计算百分比
    df['百分比'] = df['文件数量'] / total_files * 100 if total_files > 0 else 0
    
    # 按文件数量排序
    df = df.sort_values(by='文件数量', ascending=False)
    
    # 打印数据框以进行调试
    print(df)
    
    # 获取根目录文件夹名称
    root_dir_name = os.path.basename(os.path.normpath(main_dir))
    
    # 条形图
    fig_bar = px.bar(
        df, 
        x='类别', 
        y='文件数量',
        title=f'各类别的文件数量分布 ({root_dir_name})',  # 更新标题以显示根目录文件夹名称
        text=df['文件数量'],
        color='文件数量',
        color_continuous_scale='Viridis'
    )
    
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig_bar.update_layout(xaxis_tickangle=-45)
    
    fig_bar.show()
    
    # 饼图
    fig_pie = px.pie(
        df, 
        names='类别', 
        values='文件数量',
        title=f'各类别的文件数量占比 ({root_dir_name})',  # 更新标题以显示根目录文件夹名称
        hover_data=['百分比'],
        labels={'百分比':'百分比'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.show()
