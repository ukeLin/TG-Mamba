import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms, models
from my_dataset import MyDataSet
from MedMamba import VSSM as create_model
from vmamba import Backbone_VSSM as cm
from utils import read_split_data, create_lr_scheduler, train_one_epoch, evaluate, plot_roc_curve, plot_pr_curve
import datetime
import torch.nn as nn
import types
import numpy as np
import random

# 导入三个对比模型
from CellViT.models.segmentation.cell_segmentation.cellvit import CellViT, CellViTSAM
from CTransCNN.model.models.backbones.CTransCNN import my_hybird_CTransCNN
from CTransCNN.model.models.heads.my_hybird_head import My_Hybird_Head
from MedViT.MedViT import MedViT_small

def load_model(model_name, num_classes, device, pretrained_weights=None, cellvit_weights=None, medvit_weights=None):
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
    # CellViT-SAM (Base version)
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
        
        # Load CellViT pretrained weights if available
        if cellvit_weights:
            print(f"Loading CellViT pretrained weights from: {cellvit_weights}")
            state_dict = torch.load(cellvit_weights, map_location='cpu')
            # Try to load state dict, ignoring mismatched keys
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded CellViT pretrained weights: {msg}")
    # CTransCNN (Base version)
    elif model_name == 'CTransCNN-B':
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
    # MedViT (Small version)
    elif model_name == 'MedViT-S':
        model = MedViT_small(
            num_classes=num_classes,
            pretrained=True  # We'll load weights manually
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = model.to(device)
    return model


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    
    # Set workers init seed
    def seed_worker(worker_id):
        worker_seed = 42
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, train_images_bag_id, val_images_path, val_images_label, val_images_bag_id = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              images_bag_id=train_images_bag_id,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            images_bag_id=val_images_bag_id,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count() or 1, batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               worker_init_fn=seed_worker,
                                               generator=g,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             collate_fn=val_dataset.collate_fn)

    # 定义多个模型
    model_names = [
        # 'VGG19', 
        # 'ResNet34', 'ResNet50', 'DenseNet', 'EfficientNet', 
        # 'Swin-T', # 传统CNN模型
        # 'Swin-S',
        # 'Swin-B',  # Swin Transformer
        # 'ConvNeXt-B',  # ConvNeXt
        # 'ConvNeXt-S',
        # 'ConvNeXt-T',
        # 'VMamba',  # VMamba模型
        # 'CellViT',  # Changed from 'CellViT-SAM' to 'CellViT'
        # 'CTransCNN-B',  # CTransCNN
        # 'MedViT-S',  # MedViT Small version
        # 'TG-Mamba(Our)'  # 我们的模型
    ]
    models = {name: load_model(name, args.num_classes, device, args.weights, args.cellvit_weights, args.medvit_weights) for name in model_names}
    TG_Mamba = create_model(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=args.num_classes).to(device)
    models['TG-Mamba(Our)'] = TG_Mamba
    # 在 main 函数开头，初始化存储概率的字典
    model_probabilities = {name: [] for name in model_names + ['TG-Mamba(Our)']}
    #       model_probabilities = {name: [] for name in model_names}
    for model_name, model in models.items():
        print("current_model:", model_name)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)
        best_acc = 0.
        patience = 10  # 早停的耐心值，即在验证集上没有提升的epoch数
        no_improvement_count = 0  # 记录没有提升的epoch数

        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler,
                                                    excel_root=args.excel_root,
                                                    model_name=model_name)

            # validate
            val_loss, val_acc, precision, recall, F1_score, val_labels, val_probs = evaluate(model=model,
                                                                                             data_loader=val_loader,
                                                                                             device=device,
                                                                                             epoch=epoch,
                                                                                             excel_root=args.excel_root,
                                                                                             model_name=model_name)
            # 存储每个模型的预测概率和真实标签
            model_probabilities[model_name] = val_probs  # 重置概率列表
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            with open(results_file, "a") as f:
                train_info = f"[model: {model_name}, epoch: {epoch}]\n" \
                             f"train_loss: {train_loss:.4f}\n" \
                             f"train_acc: {train_acc:.4f}\n" \
                             f"val_loss: {val_loss:.4f}\n" \
                             f"val_acc: {val_acc:.4f}\n" \
                             f"precision: {precision:.4f}\n" \
                             f"recall: {recall:.4f}\n" \
                             f"F1_score: {F1_score:.4f}\n" \
                             f"lr: {optimizer.param_groups[0]['lr']:.6f}\n"

                f.write(train_info + "\n\n")
            tb_writer.add_scalar(f"{model_name}/train_loss", train_loss, epoch)
            tb_writer.add_scalar(f"{model_name}/train_acc", train_acc, epoch)
            tb_writer.add_scalar(f"{model_name}/val_loss", val_loss, epoch)
            tb_writer.add_scalar(f"{model_name}/val_acc", val_acc, epoch)
            tb_writer.add_scalar(f"{model_name}/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            # 早停机制
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"./weights/{model_name}_best_model.pth")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping for {model_name} at epoch {epoch} as no improvement in {patience} epochs.")
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--excel_root', type=str, default="114514.xlsx")
    parser.add_argument('--data-path', type=str, default=r"./rebuild(lung)")
    parser.add_argument('--weights', type=str, default='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
                        help='initial weights path')
    parser.add_argument('--cellvit-weights', type=str, default='./CellViT-256-x40.pth',
                        help='Path to CellViT pretrained weights')
    parser.add_argument('--medvit-weights', type=str, default='./MedViT_base_im1k.pth',
                        help='Path to MedViT pretrained weights')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
