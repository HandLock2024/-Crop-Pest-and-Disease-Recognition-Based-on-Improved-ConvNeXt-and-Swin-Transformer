import os
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import CombinedModel  # 从 model.py 导入定义好的 CombinedModel
from LossFunction import FocalLoss
import numpy as np
from sklearn.metrics import accuracy_score

# Mixup 数据增强函数
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 计算混合后的损失
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 训练模型函数（加入 Mixup）
# 修改后的 train_model 函数
def train_model(net, train_loader, criterion, optimizer, device, mixup_alpha=1.0):
    net.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Mixup 数据增强
        images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        optimizer.zero_grad()  # 清除累积的梯度
        outputs = net(images)  # 前向传播
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item())
    
    epoch_time = time.time() - start_time  # 计算一个epoch的时间
    epoch_loss = running_loss / len(train_loader)  # 平均损失
    epoch_accuracy = 100 * correct / total  # 准确率

    return epoch_loss, epoch_accuracy, epoch_time


# 验证模型函数
def validate_model(net, val_loader, criterion, device):
    net.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

# 动态调整学习率
def get_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 计算类别频率的函数
def calculate_class_frequencies(data_loader, num_classes):
    class_counts = np.zeros(num_classes)
    total_samples = 0

    for _, labels in data_loader:
        labels = labels.cpu().numpy()  # 将标签转换为 NumPy 数组
        for label in labels:
            class_counts[label] += 1
        total_samples += len(labels)

    # 计算每个类别的频率
    class_frequencies = class_counts / total_samples
    return class_frequencies

# 加载模型权重
def load_model_weights(model, weights_path, device, freeze_layers=False):
    if weights_path != "":
        assert os.path.exists(weights_path), f"Weights file: '{weights_path}' does not exist."
        weights_dict = torch.load(weights_path, map_location=device)
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(weights_dict, strict=False)
    if freeze_layers:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
            else:
                print(f"Training {name}")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    batch_size = 64
    num_classes = 39
    epochs = 50  # 总共训练的 epoch 数

    # 数据集转换
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(224 * 1.143)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), ''))
    image_path = os.path.join(data_root, "PV")
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化 CombinedModel
    combined_model = CombinedModel(num_classes=num_classes).to(device)

    # 加载预训练权重
    convnext_weights_path = "/home/ubuntu/ST/Mymodel/MY/convnext_small_1k_224_ema.pth"
    swin_weights_path = "/home/ubuntu/ST/Mymodel/MY/swin_small_patch4_window7_224.pth"

    combined_model.convnext = load_model_weights(combined_model.convnext, convnext_weights_path, device)
    combined_model.swin_transformer = load_model_weights(combined_model.swin_transformer, swin_weights_path, device)

    # 计算类别频率
    class_frequencies = calculate_class_frequencies(train_loader, num_classes)
    category_freq = torch.tensor(class_frequencies, device=device)  # 转为 PyTorch 张量

    # 定义损失函数和优化器
    criterion = FocalLoss(num_classes=num_classes,alpha=1, gamma=2, reduction='mean')
    optimizer = optim.AdamW(combined_model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer)  # 动态调整学习率
    best_val_acc = 0
    # 开始训练
    for epoch in range(epochs):
        # 训练模型，返回训练损失、训练准确率和所花时间
        train_loss, train_acc, train_time = train_model(combined_model, train_loader, criterion, optimizer, device)

        # 验 证模型，返回验证损失和验证准确率
        val_loss, val_acc = validate_model(combined_model, validate_loader, criterion, device)

        # 打印每轮结果
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}%, Time: {train_time:.2f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(combined_model.state_dict(), "best_model_weights.pth")
            print(f"Best model updated at epoch {epoch + 1}, Val Accuracy: {val_acc:.4f}%")

        # 使用验证损失动态调整学习率
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
