import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 如果 alpha 是一个数值，将它扩展为类别数量的列表
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * num_classes)
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        # inputs 为模型预测的logits，大小为 (batch_size, num_classes)
        # targets 为真实标签，大小为 (batch_size) (每个样本的标签是类别的索引)

        # 转换 logits 为概率
        logpt = F.log_softmax(inputs, dim=1)  # 对输入进行 softmax 再取 log
        pt = torch.exp(logpt)  # 得到概率

        # 根据 targets 选择对应的概率
        logpt = logpt.gather(1, targets.unsqueeze(1)).view(-1)  # 获取正确类别的 log概率
        pt = pt.gather(1, targets.unsqueeze(1)).view(-1)  # 获取正确类别的概率

        # 如果 alpha 是可变的，用 alpha 进行类别加权
        if self.alpha is not None:
            # 将 alpha 转移到与 targets 相同的设备上
            at = self.alpha.to(targets.device)[targets]
            logpt = logpt * at

        # 计算 Focal Loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # 根据 reduction 的设置返回损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class EqualizationLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', freq_threshold=0.01, two_stage=False):
        """
        alpha: 缩放因子，用来调节难分类样本的权重，默认为1.0
        gamma: 控制惩罚力度，默认为2.0
        reduction: 定义损失的返回方式：'none' | 'mean' | 'sum'
        freq_threshold: 用于判断类别是否为长尾类别的频率阈值
        two_stage: 用于控制是否处于两阶段训练的特征提取阶段
        """
        super(EqualizationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.freq_threshold = freq_threshold
        self.two_stage = two_stage  # 控制是否为特征提取阶段

    def forward(self, logits, target, category_freq=None):
        """
        logits: [batch_size, num_classes] 模型输出
        target: [batch_size] 目标类别
        category_freq: [num_classes] 每个类别的频率
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # 计算log_softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, num_classes]
        
        # 获取目标类别的log概率值
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # 计算概率值
        probs = torch.exp(target_log_probs)  # [batch_size]

        # 计算focal weight
        focal_weight = (1 - probs).pow(self.gamma)  # [batch_size]

        # 如果是特征提取阶段，则忽略 category_freq（即不考虑类别频率）
        if self.two_stage or category_freq is None:
            eql_loss = -self.alpha * focal_weight * target_log_probs  # 只使用 focal weight
        else:
            # 确定目标类别的频率
            target_freq = category_freq[target]  # [batch_size]

            # 如果类别的频率低于阈值，忽略负梯度（即常见类别的影响）
            eql_weight = torch.where(target_freq < self.freq_threshold, focal_weight, torch.ones_like(focal_weight))

            # 计算 Equalization Loss
            eql_loss = -self.alpha * eql_weight * target_log_probs  # [batch_size]

        # 返回损失值
        if self.reduction == 'mean':
            return eql_loss.mean()
        elif self.reduction == 'sum':
            return eql_loss.sum()
        else:
            return eql_loss
