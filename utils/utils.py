import os
import random
import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np
import torch
import random
import scipy.io as scio
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
import collections
from skimage.metrics import structural_similarity
from peft import get_peft_model_state_dict
from scipy.signal import resample, hilbert2
import math
import torch.nn as nn
import torch
import einops
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer


class CustomLoraLayer(LoraLayer, nn.Module):
    def __init__(self, r, lora_alpha, lora_dropout, in_features, out_features):
        nn.Module.__init__(self)  # 初始化 nn.Module
        LoraLayer.__init__(self, r)  # 初始化 LoraLayer

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))  # r × in_features
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # out_features × r

        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 1×1 卷积
        # 1×1 卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(
            in_channels=in_features,  # 输入通道固定为 1，因为输入被 reshape 为 [batch_size, 1, seq_len, hidden_dim]
            out_channels=out_features,  # 输出通道同样为 1
            kernel_size=1   # 1x1 卷积核
            ),
            nn.ReLU())

        # 为兼容性添加虚拟权重
        self.register_buffer("weight", torch.empty(0))  # 非可训练的属性

    def forward(self, x):
        # LoRA 映射
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T


        # 1×1 卷积操作
        batch_size, seq_len, hidden_dim = x.size()
        x =einops.rearrange(x, "b l d -> b d l")
        conv_out = self.conv1x1(x)  # [batch_size, 1, seq_len, hidden_dim]
        conv_out = einops.rearrange(conv_out, "b d l -> b l d")
        # 合并 LoRA 和卷积的输出
        return conv_out + lora_out


# 替换目标模块
def replace_lora_layer_with_custom(model, peft_config):
    """
    替换模型中的默认 LoRA 层为自定义的 CustomLoraLayer。
    """
    for name, module in model.named_modules():
        # 检查是否包含目标模块名称
        if any(target in name for target in peft_config.target_modules):
            # 获取模块的输入输出特征维度
            if isinstance(module, nn.Linear):  # 确保是线性层
                in_features = module.in_features
                out_features = module.out_features

                # 替换为 CustomLoraLayer
                custom_lora_layer = CustomLoraLayer(
                    r=peft_config.r,
                    lora_alpha=peft_config.lora_alpha,
                    lora_dropout=peft_config.lora_dropout,
                    in_features=in_features,
                    out_features=out_features,
                )

                # 根据完整路径替换模块
                parent_name = ".".join(name.split(".")[:-1])  # 父模块名称
                child_name = name.split(".")[-1]  # 当前模块名称
                parent_module = model.get_submodule(parent_name)  # 获取父模块
                setattr(parent_module, child_name, custom_lora_layer)  # 替换模块





class GetData(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = scio.loadmat(data_path)['data37']
        self.label = scio.loadmat(data_path)['label']
        # permutation = np.random.permutation(self.data37.shape[0])#以下两步进行了数据的随机排序，即数据预处理的过程
        # self.labels = self.labels[permutation]
        # self.data37 = self.data37[permutation, :]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.tensor((self.data[item]), dtype=torch.float32), torch.tensor(
            self.label[item], dtype=torch.long)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSVStats(object):
    def __init__(self,save_dir=None):
        self.acc_train = []
        self.acc_val = []
        self.loss_train = []
        self.loss_val = []
        self.lr = []
        if save_dir is None:
            save_dir="./runs"
        self.save_dir=save_dir


    def add(self, p5_train, p5_val, l_train, l_val, train_lr):
        self.acc_train.append(p5_train)
        self.acc_val.append(p5_val)
        self.loss_train.append(l_train)
        self.loss_val.append(l_val)
        self.lr.append(train_lr)

    def write(self, patience, wait, choose, name,seed,few_shotnum):
        out = os.path.join(self.save_dir,"{}_stats_patience{}_wait{}_{}.csv".format(name, patience, wait, choose,seed,few_shotnum))
        dir = self.save_dir
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        with open(out, "w") as f:
            f.write('acc_train,acc_val,loss_train,loss_val,train_lr\n')
            for i in range(len(self.acc_val)):
                f.write("{:.5f},{:.5f},{},{},{}\n".format(
                    self.acc_train[i], self.acc_val[i],
                    self.loss_train[i], self.loss_val[i], self.lr[i]))

    def read(self, out):
        raise Exception("Unimplemented")



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, wait, choose, is_LORA, patience=7, verbose=True, delta=0, best_score=None,
                 save_best=False):
        """
        Args:
            save_path : The saved model path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_score = best_score
        self.flag = False
        self.choose = choose
        self.save_best = save_best
        self.is_LORA = is_LORA

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_loss, model):

        score = -val_loss
        if math.isnan(val_loss):
            pass
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.flag = True



    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        # 打印验证损失减少的提示信息
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 根据 `self.save_best` 选择保存路径
        if not self.save_best:
            path = os.path.join(self.save_path, '{}_best_network_loss_{}.pth'.format(self.choose, -self.best_score))
        else:
            path = os.path.join(self.save_path, 'best_network.pth')

        # 获取模型的完整 state_dict
        model_state_dict = model.state_dict()

        # 创建一个 OrderedDict 来保存需要保存的权重
        new_state_dict = collections.OrderedDict()

        # 判断模型是否被 DataParallel 包装
        if isinstance(model, torch.nn.DataParallel):
            # 如果是 DataParallel 模型，访问 `module` 下的属性
            is_lora = model.module.is_LORA
            llm_model = model.module.llm_model
        else:
            # 直接访问单卡模型的属性
            is_lora = model.is_LORA
            llm_model = model.llm_model

        if is_lora:
            # 如果使用 LoRA，获取 LoRA 的 state_dict
            lora_state_dict = get_peft_model_state_dict(llm_model)

            # 遍历原始的 state_dict，将非 `llm_model` 的参数保存
            for k, v in model_state_dict.items():
                if 'llm_model' not in k:
                    new_state_dict[k] = v

            # 将 LoRA 参数添加到 new_state_dict 中，使用 "llm_model.lora." 作为前缀
            for k, v in lora_state_dict.items():
                new_state_dict[f'llm_model.lora.{k}'] = v
        else:
            # 如果不使用 LoRA，按照原来的方式保存
            for k, v in model_state_dict.items():
                if 'llm_model' not in k:
                    new_state_dict[k] = v

        # 保存新的 state_dict
        torch.save(new_state_dict, path)

        # 更新最小验证损失
        self.val_loss_min = val_loss

        if self.verbose:
            print(f'Model saved successfully at {path}')
            
class EarlyStoppingNoLora:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, wait, choose, is_LORA, patience=7, verbose=True, delta=0, best_score=None,
                 save_best=False):
        """
        Args:
            save_path : The saved model path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_score = best_score
        self.flag = False
        self.choose = choose
        self.save_best = save_best
        self.is_LORA = is_LORA

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_loss, model):

        score = -val_loss
        if math.isnan(val_loss):
            pass
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.flag = True



    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        # 打印验证损失减少的提示信息
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 根据 `self.save_best` 选择保存路径
        if not self.save_best:
            path = os.path.join(self.save_path, '{}_best_network_loss_{}.pth'.format(self.choose, -self.best_score))
        else:
            path = os.path.join(self.save_path, 'best_network.pth')

        # 获取模型的完整 state_dict
        model_state_dict = model.state_dict()

        # 创建一个 OrderedDict 来保存需要保存的权重
        # new_state_dict = collections.OrderedDict()

        # 判断模型是否被 DataParallel 包装
        # if isinstance(model, torch.nn.DataParallel):
        #     # 如果是 DataParallel 模型，访问 `module` 下的属性
        #     # is_lora = model.module.is_LORA
        #     llm_model = model.module.llm_model
        # else:
        #     # 直接访问单卡模型的属性
        #     is_lora = model.is_LORA
        #     llm_model = model.llm_model

        # if is_lora:
        #     # 如果使用 LoRA，获取 LoRA 的 state_dict
        #     lora_state_dict = get_peft_model_state_dict(llm_model)

        #     # 遍历原始的 state_dict，将非 `llm_model` 的参数保存
        #     for k, v in model_state_dict.items():
        #         if 'llm_model' not in k:
        #             new_state_dict[k] = v

        #     # 将 LoRA 参数添加到 new_state_dict 中，使用 "llm_model.lora." 作为前缀
        #     for k, v in lora_state_dict.items():
        #         new_state_dict[f'llm_model.lora.{k}'] = v
        # else:
            # 如果不使用 LoRA，按照原来的方式保存
        # for k, v in model_state_dict.items():
        #     if 'llm_model' not in k:
        #         new_state_dict[k] = v

        # 保存新的 state_dict
        torch.save(model_state_dict, path)

        # 更新最小验证损失
        self.val_loss_min = val_loss

        if self.verbose:
            print(f'Model saved successfully at {path}')

class EarlyStopping_acc:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, wait, choose, patience=7, verbose=True, delta=0, best_score=None):
        """
        Args:
            save_path : The saved model path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = best_score
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.flag = False
        self.choose = choose

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            self.flag = True

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '{}_best_network_acc_{}.pth'.format(self.choose, self.best_score))
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc
class EarlyStoppingTra:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, wait, choose, is_LORA, patience=7, verbose=True, delta=0, best_score=None,
                 save_best=False):
        """
        Args:
            save_path : The saved model path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_score = best_score
        self.flag = False
        self.choose = choose
        self.save_best = save_best
        self.is_LORA = is_LORA

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_loss, model):

        score = -val_loss
        if math.isnan(val_loss):
            pass
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.flag = True
            
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        # 打印验证损失减少的提示信息
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 根据 `self.save_best` 选择保存路径
        if not self.save_best:
            path = os.path.join(self.save_path, 'pretrain_best_network.pth')
        else:
            path = os.path.join(self.save_path, 'pretrain_best_network.pth')

        # 获取模型的完整 state_dict
        model_state_dict = model.state_dict()

        # 创建一个 OrderedDict 来保存需要保存的权重
        new_state_dict = collections.OrderedDict()
    
        # 保存新的 state_dict
        torch.save(new_state_dict, path)

        # 更新最小验证损失
        self.val_loss_min = val_loss

        if self.verbose:
            print(f'Model saved successfully at {path}')

def adjust_learning_rate(optimizer, lr, declay=0.5):
    """Sets the learning rate when we have to"""

    lr = lr * declay
    print("Learning rate: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def random_split_dataset(path, split_ratio):
    """
    将数据和标签按样本维度B随机划分

    Args:
        path (str): 数据文件路径
        split_ratio (float): 划分比例, 介于 0 ~ 1 之间

    Returns:
        tuple: (split_data, split_labels), 分别对应划分后的数据和标签
               如果 split_ratio=0，则返回 (None, None)
               如果 split_ratio=1，则返回整个数据集
    """
    np.random.seed(None)
    random.seed(None)
    data = []
    labels = []
    if split_ratio == 0:
        return (None, None)
    if path.endswith('.mat'):
        data.append(scio.loadmat(path)['data'])
        labels.append(scio.loadmat(path)['label'].flatten())
    
    elif path.endswith('.npy'):
        data.append(np.load(path))
        labels.append(np.load(path.replace('_data', '_label')))
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            # 从文件中反序列化出数据字典
            dataset = pickle.load(f)
        
        # 提取出data和label数组
        data.append(dataset['data'])
        labels.append(dataset['label'])

    data = np.concatenate(data)
    print('conc OK')
    labels = np.concatenate(labels).flatten()
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same batch size"

    if split_ratio == 1:
        return data, labels

    assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1"

    # 计算划分索引
    num_samples = data.shape[0]
    split_idx = int(num_samples * split_ratio)

    # 沿第0维(B)随机打乱数据和标签的顺序
    perm = np.random.permutation(num_samples)
    data = data[perm]
    labels = labels[perm]
   

    # 划分数据和标签
    split_data = data[:split_idx]
    split_labels = labels[:split_idx]

    return split_data, split_labels


def split_dataset(valsplit, train_set):
    train_set, val_set = torch.utils.data.random_split(train_set, [int(len(train_set) * valsplit),
                                                                   int(len(train_set)) - int(
                                                                       len(train_set) * valsplit)])
    return train_set, val_set


def calculate_mask_ratio(mask):
    """
    计算每个样本的mask_ratio

    Args:
        mask (torch.Tensor): 形状为(B, 2, 1000)的mask矩阵,其中1代表被mask

    Returns:
        mask_ratio (torch.Tensor): 形状为(B, 1)的矩阵,每个元素表示对应样本的mask_ratio
    """
    # 首先将mask从(B, 2, 1000)重塑为(B, 2000)
    mask = mask.view(mask.size(0), -1)

    # 对每个样本计算mask的元素和,得到mask的总数
    mask_sum = mask.sum(dim=1, keepdim=True)

    # 计算mask_ratio
    mask_ratio = mask_sum.float() / mask.size(1)

    # 保留两位小数
    mask_ratio = torch.round(mask_ratio * 100) / 100

    return mask_ratio


def maxminnorm(yuantun):
    x = (yuantun - np.min(yuantun)) / (np.max(yuantun) - np.min(yuantun) + 1e-6)
    return x


def closest_factors(X):
    sqrt_X = int(math.isqrt(X))  # 求出平方根的整数部分

    # 初始化最接近的两个因子
    L, W = sqrt_X, sqrt_X

    # 寻找更接近的两个因子
    while True:
        product = L * W
        if product == X:
            return L, W
        elif product < X:
            W += 1
        else:
            W -= 1
            L += 1
        # 防止无限循环
        if L > X // 2:
            break
    # 无法找到完全相等的两个因子,返回最接近的两个因子
    return L - 1, W


def compute_ssim(sgn_pred, sgn_c_np):
    """
    计算两个张量 sgn_pred 和 sgn_c_np 之间的平均 SSIM。

    Args:
        sgn_pred (torch.Tensor): 形状为 (B, C, L) 的张量。
        sgn_c_np (torch.Tensor): 形状为 (B, C, L) 的张量。

    Returns:
        float: 所有样本的平均 SSIM 值。
    """
    # 
    print(sgn_pred.shape, sgn_c_np.shape)

    # 确保输入是 NumPy 数组
    if isinstance(sgn_pred, torch.Tensor):
        sgn_pred = sgn_pred.detach().cpu().numpy()
    if isinstance(sgn_c_np, torch.Tensor):
        sgn_c_np = sgn_c_np.detach().cpu().numpy()
        
    print(sgn_pred.shape, sgn_c_np.shape)
    L, W = closest_factors(2 * sgn_pred.shape[2])
    ssim_vals = []
    for i in range(sgn_pred.shape[0]):
        # pred_sample = maxminnorm(sgn_pred[i].detach().cpu().numpy())
        # c_np_sample = maxminnorm(sgn_c_np[i].detach().cpu().numpy())
        pred_sample = maxminnorm(sgn_pred[i])  # 归一化
        c_np_sample = maxminnorm(sgn_c_np[i])  # 归一化

        pred_sample = pred_sample.reshape(L, W)
        c_np_sample = c_np_sample.reshape(L, W)
        ssim_val = structural_similarity(pred_sample, c_np_sample, data_range=1)
        ssim_vals.append(ssim_val)
    ssim_vals = np.array(ssim_vals)
    mean_ssim = np.mean(ssim_vals)
    return mean_ssim


def toIQ(sgn):
    newsgn = np.zeros((2, sgn.shape[0]))
    y = hilbert2(sgn)
    newsgn[0] = np.real(y)
    newsgn[1] = np.imag(y)
    return newsgn


def prepare_data(yaml_config, split_ratios, Names, data_paths):
    # thresholds = [yaml_config[name]['threshold'] for name in Names]
    thresholds = {name: np.float32(yaml_config[name]['threshold']) for name in Names}

    balence = [yaml_config[name]['balence'] for name in Names]

    # split_ratios = [1.0, 0.0, 0.00]
    batch_sizes = [yaml_config[name]['batchsize'] for name in Names]
    num_workers = [yaml_config[name]['numworks'] for name in Names]

    data_sets = [random_split_dataset(path, split_ratio) for path, split_ratio in zip(data_paths, split_ratios)]
    print('split load OK')
    # 获取split_ratios中值为0的索引
    zero_indices = [i for i, ratio in enumerate(split_ratios) if ratio == 0]

    # 创建一个新的列表,删除data_sets中对应索引的元素
    # data_sets = [data_sets[i] for i in range(len(data_sets)) if i not in zero_indices]
    zero_indices_names = [i for i, x in enumerate(data_sets) if i in zero_indices]
    Names = [n for i, n in enumerate(Names) if i not in zero_indices_names]
    batch_sizes = [n for i, n in enumerate(batch_sizes) if i not in zero_indices_names]
    num_workers = [n for i, n in enumerate(num_workers) if i not in zero_indices_names]
    balence = [n for i, n in enumerate(balence) if i not in zero_indices_names]
    # thresholds = [n for i, n in enumerate(thresholds) if i not in zero_indices_names]
    data_sets = [data_sets[i] for i in range(len(data_sets)) if i not in zero_indices]
    data_list, label_list = list(zip(*data_sets))
    return Names, batch_sizes, num_workers, balence, thresholds, data_list, label_list


def mixup(sgn_c, low=0.8, high=0.9):
    # 确保 alpha 在 [0.1, 0.9] 之间
    alpha = random.uniform(low, high)
    index = torch.randperm(sgn_c.size(0))
    output = alpha * sgn_c + (1 - alpha) * sgn_c[index, :]
    return output


def add_module_prefix(pretrained_dict):
    """
    将不以 'module.' 开头的键名加上 'module.' 前缀。

    Args:
        pretrained_dict (dict): PyTorch 权重字典。

    Returns:
        dict: 修改后的权重字典。
    """
    new_dict = {}

    for key, value in pretrained_dict.items():
        # 如果 key 不以 'module.' 开头，就在前面加上 'module.'
        if not key.startswith('module.'):
            new_key = 'module.' + key
        else:
            new_key = key
        # 更新新的字典
        new_dict[new_key] = value

    return new_dict


def load_dict(model, model_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}

    # Determine if 'module.' prefix is used in model keys
    model_keys = list(model_dict.keys())
    model_is_dataparallel = any(key.startswith('module.') for key in model_keys)
    # print(model_keys)

    # Determine if 'module.' prefix is used in checkpoint keys
    pretrained_keys = list(pretrained_dict.keys())

    weights_are_dataparallel = any(key.startswith('module.') for key in pretrained_keys)
    print("model_is_dataparallel", model_is_dataparallel)
    print("weights_are_dataparallel", weights_are_dataparallel)
    if isinstance(model, torch.nn.DataParallel):
        # 如果是 DataParallel 模型，访问 `module` 下的属性
        is_lora = model.module.is_LORA
        llm_model = model.module.llm_model
    else:
        # 直接访问单卡模型的属性
        is_lora = model.is_LORA
        llm_model = model.llm_model

    # Function to adjust checkpoint keys to match model keys
    # print("pretrained:", pretrained_keys)
    def adjust_key(key):
        if key.startswith('llm_model.lora'):
            # print(key)
            cur = key[len("llm_model.lora."):]
            key = "llm_model." + cur

            if 'lora_' in key and 'default' not in key:
                parts = key.split('.')
                # 在适当的位置插入 'default'
                if 'conv1x1' not in key:
                    parts.insert(-1, 'default')  # 假设 'default' 应该在 'lora_A' 之前
                else:
                    parts.insert(-3, 'default')
                key = '.'.join(parts)
            # print(key)
            return key

        new_key = key
        # return cur_key
        if weights_are_dataparallel and model_is_dataparallel:
            return new_key
        # Remove 'module.' prefix if present in checkpoint but not in model
        elif not weights_are_dataparallel and not model_is_dataparallel:
            return new_key
        elif weights_are_dataparallel and not model_is_dataparallel:
            # print("delete module")
            if key.startswith('module.'):
                new_key = key[len('module.'):]
            return new_key
        # Add 'module.' prefix if present in model but not in checkpoint
        elif not weights_are_dataparallel and model_is_dataparallel:
            # print("add  module")
            if not key.startswith('module.'):
                new_key = 'module.' + key
            return new_key

    # Adjust the keys in the checkpoint
    adjusted_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        adjusted_k = adjust_key(k)
        adjusted_pretrained_dict[adjusted_k] = v

    if model_is_dataparallel:
        adjusted_pretrained_dict = add_module_prefix(adjusted_pretrained_dict)
    # Filter out unnecessary keys and ensure shapes match
    for k, v in adjusted_pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print('Load success')

    if no_load_key:
        print("Keys that were not loaded:", no_load_key)
    else:
        print("All keys were loaded successfully.")

    return model

    
class MergedLoader:
    def __init__(self, loaders):
        self.loaders = [iter(loader) for loader in loaders]
        self.loader_datasets = [loader.dataset for loader in loaders]
        self.lengths = [len(loader) for loader in loaders]
        self.total_length = sum(self.lengths)
        self.probabilities = [l / self.total_length for l in self.lengths]
        self.active_loaders = list(range(len(loaders)))  # 记录当前还有数据的loader索引

    def __iter__(self):
        return self
    
    def __next__(self):
        while self.active_loaders:
            # 根据概率选择一个还有数据的 loader
            i = random.choices(self.active_loaders, weights=[self.probabilities[j] for j in self.active_loaders])[0]
            loader_iter = self.loaders[i]

            try:
                batch = next(loader_iter)
                return batch, i
            except StopIteration:
                # 如果这个 loader 已经遍历完, 不再使用它
                self.active_loaders.remove(i)

        # 如果所有的 loader 都已经用尽
        raise StopIteration("All loaders exhausted in this epoch.")