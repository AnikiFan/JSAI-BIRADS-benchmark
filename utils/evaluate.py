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
    figsize (tuple): 图像大小（未使用，但保留参数以保持接口一致）。
    
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

if __name__ == "__main__":
    csv_file = 'predictions.csv'  # 请将此处替换为您的CSV文件路径
    evaluate_predictions(csv_file)