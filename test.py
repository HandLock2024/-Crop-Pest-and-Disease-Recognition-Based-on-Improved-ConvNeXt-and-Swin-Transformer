import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import CombinedModel  # 从model.py导入定义好的CombinedModel
from torchvision import transforms, datasets

def load_model_weights(model, weights_path, device):
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        weights_dict = torch.load(weights_path, map_location=device)
        mismatch = model.load_state_dict(weights_dict, strict=False)
        print(f"Weights loaded with the following mismatches: {mismatch}")
    else:
        raise FileNotFoundError(f"Pre-trained weights file {weights_path} not found.")
    return model

def evaluate_model(net, test_loader, device):
    net.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(labels, preds, classes, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def save_results_to_csv(metrics, filename):
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for evaluation.")

    batch_size = 64
    num_classes = 39 # 根据你的类别数量进行更新
    data_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.143)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_folder = '/home/ubuntu/ST/Mymodel/MY/PV/test'  # 确保路径正确
    assert os.path.exists(test_folder), f"{test_folder} does not exist."
    test_dataset = datasets.ImageFolder(root=test_folder, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化 CombinedModel
    combined_model = CombinedModel(num_classes=num_classes).to(device)
    
    # 加载刚刚训练中保存的最佳模型权重
    saved_model_path = 'best_model_weights.pth'  # 假设训练时保存的模型路径
    combined_model = load_model_weights(combined_model, saved_model_path, device)

    # 评估模型
    all_labels, all_preds, all_probs = evaluate_model(combined_model, test_loader, device)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC AUC (OVO): {roc_auc:.4f}")

    # 保存评估结果
    metrics = {
        'Accuracy': accuracy,
        'Recall': recall,
        'F1 Score': f1,
        'Precision': precision,
        'ROC AUC (OVO)': roc_auc
    }
    save_results_to_csv(metrics, 'evaluation_metrics.csv')
    plot_confusion_matrix(all_labels, all_preds, list(test_dataset.class_to_idx.keys()), 'confusion_matrix.png')

if __name__ == '__main__':
    main()
