import torch
import torch.nn as nn
from ConvNext import convnext_small as ConvNeXt
from SwinTransformer import swin_small_patch4_window7_224 as SwinTransformer
from CAFM import CAFMFusion  # 替换 PSFM 为 CAFMFusion

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        
        # 实例化 ConvNeXt 和 Swin Transformer
        self.convnext = ConvNeXt(num_classes=num_classes)  # 从 ConvNeXt 模块导入
        self.swin_transformer = SwinTransformer(num_classes=num_classes)  # 从 SwinTransformer 模块导入
        
        # 实例化 CAFMFusion 模块
        self.cafm_fusion = CAFMFusion(dim=768)  # 替换为 CAFMFusion，假设通道数为 1024
        
        # 最后的分类层
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # 从 ConvNeXt 提取特征，但不进行池化或分类
        convnext_features = self.extract_convnext_features(x)
        
        # 从 Swin Transformer 提取特征，但不进行池化或分类
        swin_features = self.extract_swin_features(x)
        
        # 通过 CAFMFusion 模块融合特征
        fused_features = self.cafm_fusion(convnext_features, swin_features)
        
        # 分类：使用池化后进行分类
        out = self.classifier(fused_features.mean(dim=[2, 3]))  # 全局平均池化后进行分类
        return out

    def extract_convnext_features(self, x):
        # 从 ConvNeXt 中提取到最后的卷积特征，而不经过池化和分类头
        for i in range(4):
            x = self.convnext.downsample_layers[i](x)
            x = self.convnext.stages[i](x)
        return x  # 输出为 [B, C, H, W]

    def extract_swin_features(self, x):
        # Swin Transformer 中提取到最后的卷积特征，而不经过池化和分类头
        x, H, W = self.swin_transformer.patch_embed(x)
        x = self.swin_transformer.pos_drop(x)
        
        for layer in self.swin_transformer.layers:
            x, H, W = layer(x, H, W)
        
        # 这里不进行全局池化和分类头
        return x.view(x.size(0), H, W, -1).permute(0, 3, 1, 2)  # [B, C, H, W]

