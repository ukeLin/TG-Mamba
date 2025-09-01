import os
import sys
import json
import pickle
import random
import math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
from graph import plot_matrix, plot_roc_curve, plot_pr_curve, get_model_params, get_model_flops
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import matplotlib
# matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('./class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    train_images_bag_id = []  # 存储训练集图片对应的包ID
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    val_images_bag_id = []  # 存储验证集图片对应的包ID
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        
        # 按包ID分组
        bag_groups = {}
        for img_path in images:
            # 从文件名中提取包ID
            filename = os.path.basename(img_path)
            bag_id = filename.split('_')[0]  # 获取编号部分
            if bag_id not in bag_groups:
                bag_groups[bag_id] = []
            bag_groups[bag_id].append(img_path)
        
        # print(f"\nProcessing class: {cla}")
        # print(f"Found {len(bag_groups)} bags")
        # print(f"Example bag sizes: {[len(imgs) for imgs in list(bag_groups.values())[:3]]}")
        
        # 按包为单位进行训练集和验证集的划分
        bag_ids = list(bag_groups.keys())
        val_bag_ids = random.sample(bag_ids, k=int(len(bag_ids) * val_rate))
        
        for bag_id, bag_images in bag_groups.items():
            if bag_id in val_bag_ids:  # 验证集
                val_images_path.extend(bag_images)
                val_images_label.extend([image_class] * len(bag_images))
                val_images_bag_id.extend([bag_id] * len(bag_images))
            else:  # 训练集
                train_images_path.extend(bag_images)
                train_images_label.extend([image_class] * len(bag_images))
                train_images_bag_id.extend([bag_id] * len(bag_images))

    print("\nDataset Summary:")
    print(f"Total {sum(every_class_num)} images")
    print(f"Training: {len(train_images_path)} images, {len(set(train_images_bag_id))} bags")
    print(f"Validation: {len(val_images_path)} images, {len(set(val_images_bag_id))} bags")
    
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, train_images_bag_id, val_images_path, val_images_label, val_images_bag_id


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = '../图像分类基础/ResNet/class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, excel_root, model_name):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 设置当前epoch
    if hasattr(model, 'mil_head'):
        model.mil_head.epoch = epoch

    sex_encoder = LabelEncoder()
    stage_encoder = LabelEncoder()

    df = pd.read_excel(excel_root)
    df.columns = ['id', 'age', 'Sex', 'TMB', 'Stage']
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['TMB'].fillna(df['TMB'].mean(), inplace=True)
    df['Stage'].fillna(method='bfill', inplace=True)
    df['Stage'].fillna(method='ffill', inplace=True)
    df['Sex'] = sex_encoder.fit_transform(df['Sex'])
    df['Stage'] = stage_encoder.fit_transform(df['Stage'])
    
    scaler = MinMaxScaler()
    df[['age', 'TMB']] = scaler.fit_transform(df[['age', 'TMB']])
    peoples = [i for i in df.id]
    df = df[['id', 'age', 'Sex', 'Stage']]

    for step, data in enumerate(data_loader):
        images, labels, bag_ids, paths = data
        current_batch_size = images.shape[0]  # 获取当前batch的实际大小

        # 读取患者信息
        patients_text = []
        for p in paths:
            for i in range(len(peoples)):
                if peoples[i] in p:
                    patient_info = df[df['id'] == peoples[i]].iloc[:, 1:].values.flatten()
                    patients_text.append(patient_info)
                    break

        patients_text = torch.tensor(patients_text, dtype=torch.float32)
        patients_text = patients_text.to(images.dtype)

        sample_num += images.shape[0]

        if model_name == 'TG-Mamba(Our)' or model_name == 'MedMamba':
            instance_logits, typicalities, soft_labels, bag_predictions = model(
                images.to(device),
                patients_text.to(device),
                labels.to(device)
            )
            pred_classes = torch.max(bag_predictions, dim=1)[1]
            loss = loss_function(bag_predictions, labels.to(device))
        else:
            pred = model(images.to(device))
            if isinstance(pred, dict):
                pred = pred['tissue_types']
            pred_classes = torch.max(pred, dim=1)[1]
            loss = loss_function(pred, labels.to(device))

        loss.backward()
        accu_loss += loss.detach()
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 更新进度条描述
        desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        # 如果是MIL模型，添加MIL相关信息
        if hasattr(model, 'mil_head') and hasattr(model.mil_head, 'debug_info'):
            debug = model.mil_head.debug_info
            desc += " | typ: {:.2f}±{:.2f}, key: {}+{}, conf: {:.2f}".format(
                debug['typicality_mean'],
                debug['typicality_std'],
                debug['num_pos_key'] // current_batch_size,  # 使用当前batch的实际大小
                debug['num_neg_key'] // current_batch_size,
                debug['pred_confidence']
            )

        data_loader.desc = desc
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, excel_root, model_name):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0

    # 初始化编码器和标准化器
    sex_encoder = LabelEncoder()
    stage_encoder = LabelEncoder()
    scaler = MinMaxScaler()

    df = pd.read_excel(excel_root)
    df.columns = ['id', 'age', 'Sex', 'TMB', 'Stage']
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['TMB'].fillna(df['TMB'].mean(), inplace=True)
    df['Sex'] = sex_encoder.fit_transform(df['Sex'])
    df['Stage'] = stage_encoder.fit_transform(df['Stage'])
    df[['age', 'TMB']] = scaler.fit_transform(df[['age', 'TMB']])
    peoples = [i for i in df.id]
    df = df[['id', 'age', 'Sex', 'Stage']]

    data_loader = tqdm(data_loader, file=sys.stdout)
    labels_y = []
    labels_y_hat = []
    probabilities = []  # 存储预测概率

    for step, data in enumerate(data_loader):
        images, labels, bag_ids, paths = data

        patients_text = []
        for p in paths:
            for i in range(len(peoples)):
                if peoples[i] in p:
                    patient_info = df[df['id'] == peoples[i]].iloc[:, 1:].values.flatten()
                    patients_text.append(patient_info)
                    break

        patients_text = torch.tensor(patients_text, dtype=torch.float32)
        patients_text = patients_text.to(images.dtype)

        sample_num += images.shape[0]

        if model_name == 'TG-Mamba(Our)' or model_name == 'MedMamba':
            instance_logits, typicalities, soft_labels, bag_predictions = model(
                images.to(device), 
                patients_text.to(device),
                labels.to(device)
            )
            # 使用包级别的预测计算损失和准确率
            pred_classes = torch.max(bag_predictions, dim=1)[1]
            loss = loss_function(bag_predictions, labels.to(device))
            # 存储预测概率
            probabilities.extend(torch.softmax(bag_predictions, dim=1)[:, 1].cpu().numpy())
        else:
            pred = model(images.to(device))
            if isinstance(pred, dict):
                pred = pred['tissue_types']
            pred_classes = torch.max(pred, dim=1)[1]
            loss = loss_function(pred, labels.to(device))
            probabilities.extend(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss

        # Store labels and predictions for metrics
        labels_y.extend(labels.cpu().numpy())
        labels_y_hat.extend(pred_classes.cpu().numpy())

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    if epoch == 0:
        plot_matrix(labels_y, labels_y_hat, [0, 1], "TG-Mamba", axis_labels=["High", "Low"])

    # Compute evaluation metrics
    precision = metrics.precision_score(labels_y, labels_y_hat, average='macro')
    recall = metrics.recall_score(labels_y, labels_y_hat, average='macro')
    F1 = metrics.f1_score(labels_y, labels_y_hat, average='macro')
    print(f"precision:{precision:.3f}, recall:{recall:.3f}, F1_score:{F1:.3f}")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, F1, labels_y, probabilities


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


