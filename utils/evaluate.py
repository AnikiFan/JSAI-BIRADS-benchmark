import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

def load_data(csv_path):
    """
    读取CSV文件并返回DataFrame。
    
    参数:
    csv_path (str): CSV文件路径。
    
    返回:
    pd.DataFrame: 包含预测结果的数据框。
    """
    df = pd.read_csv(csv_path)
    return df

def compute_metrics(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    计算准确率、精确率、召回率和F1分数。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    dict: 包含各项评估指标的字典。
    """
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        '准确率 (Accuracy)': accuracy,
        '精确率 (Precision)': precision,
        '召回率 (Recall)': recall,
        'F1分数 (F1-Score)': f1
    }
    
    return metrics

def get_classification_report_df(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    生成分类报告并返回为DataFrame格式。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    pd.DataFrame: 分类报告的数据框。
    """
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def plot_confusion_matrix_plotly(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    绘制混淆矩阵热力图，使用Plotly实现交互式图表。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    None
    """
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df[true_label_col].unique())
    
    # 创建一个DataFrame用于Plotly
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    fig = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(x="预测类别", y="真实类别", color="计数"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title='混淆矩阵',
        xaxis_title='预测类别',
        yaxis_title='真实类别',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )
    
    fig.show()

def plot_normalized_confusion_matrix_plotly(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    绘制归一化的混淆矩阵热力图。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    None
    """
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    labels = sorted(df[true_label_col].unique())

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig = px.imshow(
        cm_df,
        text_auto='.2f',
        labels=dict(x="预测类别", y="真实类别", color="归一化计数"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        title='归一化混淆矩阵',
        xaxis_title='预测类别',
        yaxis_title='真实类别',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )

    fig.show()

def plot_classification_metrics(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    绘制每个类别的精确率、召回率和F1分数的柱状图。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    None
    """
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    metrics = ['precision', 'recall', 'f1-score']
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=report_df.index,
            y=report_df[metric],
            name=metric
        ))
    
    fig.update_layout(
        title='每个类别的评估指标',
        xaxis_title='类别',
        yaxis_title='指标值',
        barmode='group'
    )
    
    fig.show()

def plot_class_distribution(df, true_label_col='category', pred_label_col='predicted_category'):
    """
    绘制真实类别和预测类别的分布柱状图。
    
    参数:
    df (pd.DataFrame): 包含预测结果的数据框。
    true_label_col (str): 真实标签的列名。
    pred_label_col (str): 预测标签的列名。
    
    返回:
    None
    """
    true_counts = df[true_label_col].value_counts().sort_index()
    pred_counts = df[pred_label_col].value_counts().sort_index()
    labels = sorted(set(df[true_label_col]) | set(df[pred_label_col]))

    true_values = [true_counts.get(label, 0) for label in labels]
    pred_values = [pred_counts.get(label, 0) for label in labels]

    fig = go.Figure(data=[
        go.Bar(name='真实类别', x=labels, y=true_values),
        go.Bar(name='预测类别', x=labels, y=pred_values)
    ])

    fig.update_layout(
        barmode='group',
        title='类别分布',
        xaxis_title='类别',
        yaxis_title='计数'
    )

    fig.show()

def evaluate_predictions(csv_path):
    """
    读取预测结果CSV并计算所有评估指标。
    
    参数:
    csv_path (str): CSV文件路径。
    
    返回:
    None
    """
    df = load_data(csv_path)
    
    # 计算指标
    metrics = compute_metrics(df)
    print("评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 分类报告
    report_df = get_classification_report_df(df)
    print("\n分类报告:")
    print(report_df)
    
    # 绘制混淆矩阵
    plot_confusion_matrix_plotly(df)
    
    # 绘制归一化混淆矩阵
    plot_normalized_confusion_matrix_plotly(df)
    
    # 绘制每个类别的评估指标
    plot_classification_metrics(df)
    
    # 绘制类别分布
    plot_class_distribution(df)

if __name__ == "__main__":
    csv_file = 'predictions.csv'  # 请将此处替换为您的CSV文件路径
    evaluate_predictions(csv_file)