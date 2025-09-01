import os
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from my_dataset import MyDataSet
from MedMamba import VSSM as create_model
from graph import plot_bubble_chart
from vmamba import Backbone_VSSM as cm
from utils import read_split_data, evaluate, plot_roc_curve, plot_pr_curve, get_model_params, get_model_flops
import datetime
import time
import torch.nn as nn

# 导入三个对比模型
from CellViT.models.segmentation.cell_segmentation.cellvit import CellViT, CellViTSAM
from CTransCNN.model.models.backbones.CTransCNN import my_hybird_CTransCNN
from CTransCNN.model.models.heads.my_hybird_head import My_Hybird_Head
from MedViT.MedViT import MedViT_small
def load_model(model_name, num_classes, device, weights_path=None):
    if model_name == 'ResNet34':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'VGG19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'AlexNet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'DenseNet':
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'EfficientNet':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'ViT-B':
        model = models.vit_b_16(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'ViT-H':
        model = models.vit_h_14(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'ViT-l':
        model = models.vit_l_16(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'Swin-T':
        model = models.swin_t(pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'Swin-B':
        model = models.swin_b(pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'Swin-S':
        model = models.swin_s(pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'ConvNeXt-T':
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'ConvNeXt-B':
        model = models.convnext_base(pretrained=True)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'ConvNeXt-S':
        model = models.convnext_small(pretrained=True)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'VMamba':
        model = cm(out_indices=(0, 1, 2, 3), pretrained=False, norm_layer="ln", depths=[2, 2, 4, 2], dims=96,
                   drop_path_rate=0.2, patch_size=4, in_chans=3, num_classes=2, ssm_d_state=64, ssm_ratio=1.0,
                   ssm_dt_rank="auto", ssm_act_layer="gelu", ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
                   ssm_init="v2", forward_type="m0_noz", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
                   gmlp=False, patch_norm=True, downsample_version="v3", patchembed_version="v2", use_checkpoint=False,
                   posembed=False, imgsize=224)
    elif model_name == 'VMamba-B':
        model = cm(out_indices=(0, 1, 2, 3), pretrained=False, norm_layer="ln", depths=[2, 2, 27, 2], dims=96,
                   drop_path_rate=0.6, patch_size=4, in_chans=3, num_classes=2, ssm_d_state=16, ssm_ratio=2.0,
                   ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                   ssm_init="v0", forward_type="v0", mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
                   gmlp=False, patch_norm=True, downsample_version="v3", patchembed_version="v2", use_checkpoint=False,
                   posembed=False, imgsize=224)
    elif model_name == 'VMamba-S':
        model = cm(out_indices=(0, 1, 2, 3), pretrained=False, norm_layer="ln", depths=[2, 2, 27, 2], dims=96,
                   drop_path_rate=0.3, patch_size=4, in_chans=3, num_classes=2, ssm_d_state=16, ssm_ratio=2.0,
                   ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                   ssm_init="v0", forward_type="v0", mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
                   gmlp=False, patch_norm=True, downsample_version="v3", patchembed_version="v2", use_checkpoint=False,
                   posembed=False, imgsize=224)
    elif model_name == 'CellViT':  # Changed from 'CellViT-SAM' to 'CellViT'
        model = CellViT(
            num_tissue_classes=num_classes,  # TMB classification
            embed_dim=768,  # ViT-B size
            input_channels=3,  # RGB
            depth=12,  # ViT-B depth
            num_heads=12,  # ViT-B heads
            extract_layers=[3, 6, 9, 12],  # Extract features from these layers
            drop_rate=0.1
        )
        
        # # Load CellViT pretrained weights if available
        # if cellvit_weights:
        #     print(f"Loading CellViT pretrained weights from: {cellvit_weights}")
        #     state_dict = torch.load(cellvit_weights, map_location='cpu')
        #     # Try to load state dict, ignoring mismatched keys
        #     msg = model.load_state_dict(state_dict, strict=False)
        #     print(f"Loaded CellViT pretrained weights: {msg}")
        
        # # Modify forward method to only return tissue_types
        # def new_forward(self, x):
        #     out_dict = self.old_forward(x)
        #     return out_dict['tissue_types']
        
        # # Save original forward method
        # model.old_forward = model.forward
        # # Replace with new forward method
        # model.forward = types.MethodType(new_forward, model)
    # CTransCNN (Base version)
    elif model_name == 'CTransCNN':
        backbone = my_hybird_CTransCNN(
            arch='tiny',  # 使用tiny架构，因为base架构不在arch_zoo中
            patch_size=16,
            drop_path_rate=0.2
        )
        # 获取backbone的输出特征维度
        embed_dims = backbone.embed_dims  # 通常是384
        base_channels = backbone.channel_ratio * 64  # 基础通道数
        
        # 创建分类头
        head = My_Hybird_Head(
            num_classes=num_classes,
            in_channels=[base_channels * 4, embed_dims]  # [256, 384] for tiny architecture
        )
        
        # 组合backbone和head
        model = nn.Sequential(backbone, head)
    # MedViT (Base version)
    elif model_name == 'MedViT':
        model = MedViT_small(
            num_classes=num_classes,
            pretrained=True  # We'll load weights manually
        )
        
        # # Load MedViT pretrained weights if available
        # if medvit_weights:
        #     print(f"Loading MedViT pretrained weights from: {medvit_weights}")
        #     state_dict = torch.load(medvit_weights, map_location='cpu')
        #     # Try to load state dict, ignoring mismatched keys
        #     msg = model.load_state_dict(state_dict, strict=False)
        #     print(f"Loaded MedViT pretrained weights: {msg}")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if weights_path:
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    model = model.to(device)
    return model




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 定义多个模型
    model_names = ['VGG19', 'ResNet34', 'ResNet50', 'DenseNet', 'EfficientNet', 'Swin-T', 'Swin-S', 'Swin-B', 'ConvNeXt-T', 'ConvNeXt-S', 'ConvNeXt-B',  'CellViT', 'CTransCNN', 'MedViT', 'VMamba']
    # model_names = []
    models = {}
    params = []
    flops = []
    inference_times = []
# 'VMamba',
    for name in model_names:
        weights_path = os.path.join("./weights", f"{name}_best_model.pth")
        models[name] = load_model(name, args.num_classes, device, weights_path=weights_path)

    TG_Mamba = create_model(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=args.num_classes).to(device)
    TG_Mamba_weights_path = os.path.join("./weights", "TG-Mamba(Our)_best_model.pth")
    TG_Mamba.load_state_dict(torch.load(TG_Mamba_weights_path, map_location=device))
    models['TG-Mamba(Our)'] = TG_Mamba

    # 在 main 函数开头，初始化存储概率的字典
    model_probabilities = {name: [] for name in model_names + ['TG-Mamba(Our)']}
    val_labels = []

    for model_name, model in models.items():
        print("current_model:", model_name)
        model.eval()  # 设置模型为评估模式

        # 计算模型的FLOPs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_input_size = (1, 3, img_size, img_size)
        if model_name == 'TG-Mamba(Our)':
            text_input_size = (1, 3)
        else:
            text_input_size = None
        model_flops, model_params = get_model_flops(model, image_input_size, device, text_input_size)
        flops.append(model_flops)
        params.append(model_params)

        # # 计算模型的参数量
        # param = get_model_params(model)
        # params.append(param)

        # 进行验证推理
        start_time = time.time()
        val_loss, val_acc, precision, recall, F1_score, val_labels, val_probs = evaluate(model=model,
                                                                                         data_loader=val_loader,
                                                                                         device=device,
                                                                                         epoch=0,
                                                                                         excel_root=args.excel_root,
                                                                                         model_name=model_name)
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # 存储每个模型的预测概率和真实标签
        model_probabilities[model_name] = val_probs  # 重置概率列表

        with open(results_file, "a") as f:
            val_info = f"[model: {model_name}]\n" \
                       f"val_loss: {val_loss:.4f}\n" \
                       f"val_acc: {val_acc:.4f}\n" \
                       f"precision: {precision:.4f}\n" \
                       f"recall: {recall:.4f}\n" \
                       f"F1_score: {F1_score:.4f}\n" \
                       f"Parameters: {model_params:.2f}M\n" \
                       f"FLOPs: {model_flops:.2f}G\n" \
                       f"Inference Time: {inference_time:.4f}s\n"

            f.write(val_info + "\n\n")

        tb_writer.add_scalar(f"{model_name}/val_loss", val_loss, 0)
        tb_writer.add_scalar(f"{model_name}/val_acc", val_acc, 0)

    plot_roc_curve(val_labels, [model_probabilities[name] for name in model_names + ['TG-Mamba(Our)']],
                   model_names + ['TG-Mamba(Our)'])
    plot_pr_curve(val_labels, [model_probabilities[name] for name in model_names + ['TG-Mamba(Our)']],
                  model_names + ['TG-Mamba(Our)'])
    # 在所有模型评估完成后，绘制气泡图
    plot_bubble_chart(val_labels, [model_probabilities[name] for name in model_names + ['TG-Mamba(Our)']],
                      model_names + ['TG-Mamba(Our)'], params, flops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--excel_root', type=str, default="114514.xlsx")
    parser.add_argument('--data-path', type=str, default="./rebuild(lung)")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)