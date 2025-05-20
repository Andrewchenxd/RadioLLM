import time
import matplotlib.pyplot as plt
import torch.optim
from torch.utils.data import DataLoader
from torch import nn
from dataset import *
from tqdm import tqdm
from utils.utils import *
# from utils.loss import MAELoss, InfoNCE, maeLoss, CosineSimilarityLoss
from model.radiollm import RadioLLM
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import argparse
import warnings
import copy
import torch.nn.functional as F
import yaml
import pickle
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# 检查绑定是否成功
print(f"Current process bound to CPU cores: {os.sched_getaffinity(0)}")

choose=True
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
    
    
if choose==True:
    parser = argparse.ArgumentParser(description='Train TransNet')
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--lamba", default=1) #sgn clean
    parser.add_argument("--alpha", default=1) #sgn mask
    parser.add_argument("--beta", default=0.0)  # fliter
    parser.add_argument("--delta", default=0.0)  # freq
    parser.add_argument("--balence", default=1e3)
    parser.add_argument("--parallel",default=True)
    parser.add_argument("--autocast",default=True)
    parser.add_argument("--decoder_is", type=str2bool, default=True)
    parser.add_argument("--model_name",default="RadioLLM")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=
    './checkpoint_LLM/LLM/RadioLLM/gpt2/total/xiaorong/woHTRP/TOPK7/long_term_forecast/generate/True/best_network.pth')
    parser.add_argument("--acc_it", type=int, default=32)
    parser.add_argument("--clip_grad", type=int, default=100)
    parser.add_argument('--task_name', type=str, required=False,
                        default='long_term_forecast',
                        help='task name, options: [long_term_forecast imputation, classification, soft_hard_prompt]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=128, help='prediction sequence length')
    parser.add_argument('--prompt_domain', type=str2bool, default=True)
    parser.add_argument('--content', type=str,
                        default='The RadioML 2016.10a is a comprehensive dataset for evaluating wireless signal modulation recognition algorithms. It is a vital resource in the field of cognitive radio and dynamic spectrum access, enabling efficient utilization of the electromagnetic spectrum.'
                        , help='Dataset description')

    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  #9G LLAMA, LLAMA3.2, BERT,GPT2_w/o_pretrain
    parser.add_argument('--llm_path', type=str, default='./pretrain_model/gpt2/',# gpt2
    help='LLM_model_path')
    parser.add_argument('--llm_dim', type=int, default='768',
                        help='LLM model dimension')  #  GPT2-small:768; BERT-base:768,LLAMA_3.2_1B:2048
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--right_prob', type=float, default=0.4)
    parser.add_argument('--K', type=int, default=7, help='top K prompt')
    parser.add_argument('--is_LORA', type=str2bool, default=True)
    parser.add_argument("--decode_mask", type=str2bool, default=True)
    parser.add_argument("--mix", type=str2bool, default=True)
    parser.add_argument("--attn", type=str, default='prob')
    parser.add_argument('--d_ff2', type=int, default=1024, help='dimension of fcn')

    parser.add_argument("--RGB_is", type=str2bool, default=True)
    parser.add_argument("--adsbis", type=str2bool, default=False)
    parser.add_argument("--resample", type=str2bool, default=False)
    parser.add_argument("--chazhi", type=str2bool, default=False)
    parser.add_argument("--newdata", type=str2bool, default=False)
    parser.add_argument("--cnum", type=int, default=2)
    parser.add_argument("--samplenum", type=int, default=1)  # samplenum pwvd 15 without 5
    parser.add_argument("--trans_choose", type=str, default='fft')
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--wait", type=int, default=20)
    parser.add_argument("--declay", default=0.5)
    parser.add_argument("--yuzhi", type=int, default=10)
    parser.add_argument("--pref", type=int, default=4)
    parser.add_argument("--name", type=str, default='RML2018_10a')
    parser.add_argument("--load_lora_only", type=str2bool, default=False)

    parser.add_argument('--num_class', type=int, default=11, help='num of heads')
    parser.add_argument("--torch_seed", type=str, default='42')
    parser.add_argument("--np_seed", type=str, default='None')

    parser.add_argument('--unique_keys_18', nargs='+',
                        default=['32PSK', 'OQPSK', '16APSK', 'AM-SSB-SC', 'FM', '32APSK', 'GMSK', '256QAM', 'BPSK',
                                 '64APSK', '4ASK', 'OOK', '16PSK', '64QAM', '128QAM', '8ASK', 'AM-DSB-SC', '128APSK',
                                 '8PSK', 'QPSK', '32QAM', 'AM-DSB-WC', '16QAM', 'AM-SSB-WC'],
                        help='List of unique keys for 2018 RML dataset.')
    parser.add_argument('--unique_keys_16', nargs='+',
                        default=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK',
              'WBFM'],
                        help='List of unique keys for 2018 RML dataset.')
    parser.add_argument('--Names', nargs='+', choices=['RML2016', 'RML2018', 'ADSB'], help='List of dataset names.')
    opt = parser.parse_args()

def set_random_seed(torch_seed=0,np_seed=0):
    """
    fix the random seed

    :seed: the seed want to set
    """


    torch_seed=int(torch_seed)
    if np_seed=='None':
        np_seed=None
    else:
        np_seed = int(np_seed)
    print('set random torch seed to {}'.format(torch_seed))
    print('set random numpy seed to {}'.format(np_seed))
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(np_seed)
    random.seed(np_seed)
    return torch_seed,np_seed

def acc_classes(pre, labels,BATCH_SIZE):
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc
def acc_snrs(pre, labels,snr,acc_snr_pre,acc_snr_count):
    pre_y = torch.max(pre, dim=1)[1]
    pre_y =pre_y.detach().cpu().numpy()
    labelclass=np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i]==labelclass[i]:
            acc_snr_pre[0,snr[i]]+=1
            acc_snr_count[0,snr[i]]+=1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre,acc_snr_count

def freq_norm(sgn,freq_choose='stft'):
    if freq_choose == 'fft':
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min = sgn_min.unsqueeze(1)
        sgn_max = sgn_max.unsqueeze(1)
        freq = (2 * sgn - sgn_min - sgn_max) / (sgn_max - sgn_min)
    elif freq_choose=='stft':
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_max, _ = torch.max(sgn_max, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min, _ = torch.min(sgn_min, dim=1)
        sgn_min = sgn_min.unsqueeze(1).unsqueeze(2)
        sgn_max = sgn_max.unsqueeze(1).unsqueeze(2)
        freq = (2 * sgn - sgn_min - sgn_max) / (sgn_max - sgn_min)
    return freq

def freq_resize(freq):
    freq = F.interpolate(freq.unsqueeze(1), size=(128, 128), mode='bicubic')
    freq = freq.squeeze(1)
    return freq

def phases_freq_loss(x_pred,sgn_c):
    I_pred = x_pred[0, :].squeeze(1)
    Q_pred = x_pred[1, :].squeeze(1)
    I = sgn_c[0, :].squeeze(1).to(device)
    Q = sgn_c[1, :].squeeze(1).to(device)
    phases = torch.atan2(I, Q)
    phases_pred = torch.atan2(I_pred, Q_pred)
    freq = torch.abs(torch.fft.fft(I + Q * 1j))
    freq_pred = torch.abs(torch.fft.fft(I_pred + Q_pred * 1j))
    freq = freq_norm(freq, freq_choose='fft')
    freq_pred = freq_norm(freq_pred, freq_choose='fft')
    phases_pred = freq_norm(phases_pred, freq_choose='fft')
    phases = freq_norm(phases, freq_choose='fft')
    return freq,freq_pred,phases,phases_pred

def acc_AA(pre, labels, acc_AA_pre, acc_AA_count):
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels)
    # labelclass[labelclass == 99] = 7
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_AA_pre[0, labelclass[i]] += 1
            acc_AA_count[0, labelclass[i]] += 1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre, acc_AA_count

def trainhec(train_loaders, model, criterion,criterion2,criterion3,criterion4, optimizer,
             epoch, epoch_max,scheduler,adsbis=False,prob=[],thresholds=[],balence=[],Names=[]):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis==True:
        acc_snr_pre=np.zeros((1,7))
        acc_snr_count = np.zeros((1,7))
    else:
        acc_snr_pre = np.zeros((1, 20))
        acc_snr_count = np.zeros((1, 20))
    acc_aa = np.zeros((1, 20))
    acc_oa = np.zeros((1, 20))
    # switch to train mode
    total_len=0
    merged_loader = MergedLoader(train_loaders)
    scaler = GradScaler()
    for loader in train_loaders:
        total_len+=len(loader)
    model.train()
    with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        # for i in range(len(train_loader1) + len(train_loader2)):

        for batch, i_loader in merged_loader:

            input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
            sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = input1, input2, input3, input4, input5, input6, input7, input8
            # print(sgn_a.shape)
            with autocast(enabled=args.autocast):
                if random.random() <= thresholds[Names[i_loader]][0]:
                    x_pred, mask = model(sgn_c.to(device), enable_mask=True, dataset_name=dataset_name[0])
                    x = sgn_c.to(device)
                    mask = mask.unsqueeze(1).expand_as(x)
                    loss1 =  criterion(balence[i_loader] *x_pred * mask, balence[i_loader] *x * mask)
                    freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c)
                    loss4 = criterion(phases_pred, phases)
                    loss = args.alpha * loss1 + args.delta * loss4
                    loss_mask = loss
                    losses_class1.update(loss_mask.item())
                elif random.random() <= thresholds[Names[i_loader]][1]:
                    x_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])
                    # sgn_f = torch.transpose(sgn_f, 1, 2)
                    # sgn_c = torch.transpose(sgn_c, 1, 2)
                    loss2 =  criterion(balence[i_loader] *x_pred, balence[i_loader] *sgn_f.to(device))
                    loss3 =  criterion(balence[i_loader] *x_pred, balence[i_loader] *sgn_c.to(device))

                    freq, freq_pred, phases, phases_pred=phases_freq_loss(x_pred,sgn_c)
                    loss4=criterion(phases_pred,phases)

                    loss = args.beta * loss2 + args.lamba * loss3 + args.delta * loss4
                    # loss =  args.delta * loss4
                    loss_noise = loss
                    losses_class2.update(loss_noise.item())
                elif random.random() <= thresholds[Names[i_loader]][2]:
                    sgn_m=mixup(sgn_c)
                    x_pred = model(sgn_m.to(device), enable_mask=False, dataset_name=dataset_name[0])
                    # sgn_f = torch.transpose(sgn_f, 1, 2)
                    # sgn_c = torch.transpose(sgn_c, 1, 2)
                    loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                    loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))

                    freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c)
                    loss4 = criterion(phases_pred, phases)

                    loss = args.beta * loss2 + args.lamba * loss3 + args.delta * loss4
                    # loss =  args.delta * loss4
                    loss_noise = loss
                    losses_class3.update(loss_noise.item())

            # measure accuracy and record loss
            acc.update(0)
            # 使用 scaler 缩放梯度进行反向传播
            if args.autocast:
                optimizer.zero_grad()
                scaler.scale(loss.mean()).backward()

                # 使用 scaler 调用 optimizer.step()
                scaler.step(optimizer)

                # 更新 scaler 状态
                scaler.update()
            else:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            # losses_class3.update(loss3.item())
            # print(loss)
            losses_class.update(loss.mean().item())
            scheduler.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg,
                                'acc': acc.avg})
            pbar.update(1)

    torch.cuda.empty_cache()
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)

    return acc.avg, losses_class.avg

def validatehec(val_loaders, model, criterion,criterion2,criterion3,criterion4, epoch,
                epoch_max,scheduler,adsbis=False,prob=[],thresholds=[],balence=[],Names=[]):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 7))
        acc_snr_count_val = np.zeros((1, 7))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    total_len=0
    scaler = GradScaler()
    merged_loader = MergedLoader(val_loaders)
    for loader in val_loaders:
        total_len+=len(loader)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:

            for batch, i_loader in merged_loader:

                input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
                sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = input1, input2, input3, input4, input5, input6, input7, input8
                # print(sgn_a.shape)
                with autocast(enabled=args.autocast):
                    if random.random() <= thresholds[Names[i_loader]][0]:
                        x_pred, mask = model(sgn_c.to(device), enable_mask=True, dataset_name=dataset_name[0])
                        x = sgn_c.to(device)
                        mask = mask.unsqueeze(1).expand_as(x)
                        loss1 = criterion(balence[i_loader] *x_pred * mask, balence[i_loader] * x * mask)
                        loss = args.alpha * loss1
                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c)
                        loss4 = criterion(phases_pred, phases)
                        loss_mask = loss+ args.delta * loss4
                        losses_class1.update(loss_mask.item())
                    elif random.random() <= thresholds[Names[i_loader]][1]:
                        x_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])
                        # sgn_f = torch.transpose(sgn_f, 1, 2)
                        # sgn_c = torch.transpose(sgn_c, 1, 2)
                        loss2 = criterion(balence[i_loader] *x_pred, balence[i_loader] * sgn_f.to(device))
                        loss3 = criterion(balence[i_loader] *x_pred, balence[i_loader] * sgn_c.to(device))
                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c)
                        loss4 = criterion(phases_pred, phases)
                        loss = args.beta * loss2 + args.lamba * loss3 + args.delta * loss4
                        loss_noise = loss
                        losses_class2.update(loss_noise.item())
                    elif random.random() <= thresholds[Names[i_loader]][2]:
                        sgn_m = mixup(sgn_c)
                        x_pred = model(sgn_m.to(device), enable_mask=False, dataset_name=dataset_name[0])
                        # sgn_f = torch.transpose(sgn_f, 1, 2)
                        # sgn_c = torch.transpose(sgn_c, 1, 2)
                        loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                        loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))

                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c)
                        loss4 = criterion(phases_pred, phases)

                        loss = args.beta * loss2 + args.lamba * loss3 + args.delta * loss4
                        # loss =  args.delta * loss4
                        loss_noise = loss
                        losses_class3.update(loss_noise.item())
                # measure accuracy and record loss
                acc.update(0)
                # acc_AA(output, val_label,acc_AA_pre,acc_AA_count)

                # losses_class3.update(loss3.item())
                losses_class.update(loss.mean().item())


                pbar.set_postfix(**{'val_loss_class': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)


    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)
    return acc.avg, losses_class.avg

def validatehec_ssim(val_loaders, model, criterion,criterion2,criterion3,criterion4, epoch,
                epoch_max,scheduler,adsbis=False,prob=[],thresholds=[],balence=[],Names=[]):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 7))
        acc_snr_count_val = np.zeros((1, 7))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    total_len=0
    merged_loader = MergedLoader(val_loaders)
    for loader in val_loaders:
        total_len+=len(loader)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:

            for batch, i_loader in merged_loader:

                input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
                sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = input1, input2, input3, input4, input5, input6, input7, input8
                # print(sgn_a.shape)

                x_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])
                loss = compute_ssim(x_pred, sgn_c)
                losses_class1.update(loss)
                losses_class2.update(loss)
                # measure accuracy and record loss
                acc.update(0)
                # losses_class3.update(loss3.item())
                losses_class.update(loss)
                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)


    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)
    return acc.avg, losses_class.avg

def create_folders(label,dataset_name,flag, snr):
  """
  Creates folders for label and SNR if they don't exist.

  Args:
    label: String representing the signal type (e.g., 'BPSK').
    snr: Float representing the signal-to-noise ratio.
  """
  save_dir = os.path.join('./result/TimeLLM_units/{}_soft_prompt3/{}/{}/{}/{}'.format(args.llm_model,str(dataset_name),flag,str(label), str(snr)))
  os.makedirs(save_dir, exist_ok=True)  # Create folders if they don't exist
  return save_dir

def validatehec_val(val_loader, model, criterion,criterion2,criterion3,criterion4, epoch, epoch_max,adsbis=False):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 7))
        acc_snr_count_val = np.zeros((1, 7))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    total_len = 0
    merged_loader = MergedLoader(val_loaders)
    for loader in val_loaders:
        total_len += len(loader)
    # switch to evaluate mode
    model.eval()
    plot = True
    with torch.no_grad():
        with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            i=0
            for batch, i_loader in merged_loader:
                input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
                sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = input1, input2, input3, input4, input5, input6, input7, input8
                if random.random() <=0.5:
                    flag='mask'
                    sgn_pred, mask = model(sgn_c.to(device), enable_mask=True,dataset_name=dataset_name[0])
                    mask = mask.unsqueeze(-1).transpose(2, 1)
                    mask=mask.expand_as(sgn_c).to(sgn_c.device)
                    mask_ratio = calculate_mask_ratio(mask)
                    mask=1-mask  #1 是通过，0是mask

                    sgn_n=sgn_c*mask
                else:
                    flag = 'noise'
                    sgn_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])

                sgn_pred_np=sgn_pred.detach().cpu().numpy()
                sgn_c_np = sgn_c.detach().cpu().numpy()
                sgn_n_np = sgn_n.detach().cpu().numpy()
                labels=labels.detach().cpu().numpy()
                snrs=snrs.detach().cpu().numpy()
                if flag=='mask':
                    mask_ratio=mask_ratio.detach().cpu().numpy().flatten()
                t = np.arange(0, sgn_c_np.shape[2], 1)
                if plot==True:
                    for j in range(sgn_pred_np.shape[0]):
                        label = labels[j]  # Assuming labels is a list containing signal types
                        if dataset_name[j] =='RML2016a+b_total_snr':
                            Labels=args.unique_keys_16
                            label_str = Labels[np.int16(label)]
                        elif dataset_name[j] =='RML2018_high_snr':
                            Labels = args.unique_keys_18
                            label_str = Labels[np.int16(label)]
                        elif dataset_name[j] =='ADSB':
                            label_str = str(int(label))
                        elif dataset_name[j] == 'WIFI_2ft':
                            label_str = str(int(label))
                        snr_val = snrs[j]
                        if flag == 'noise':
                            save_dir = create_folders(label_str,dataset_name[j],flag, snr_val)
                        elif flag == 'mask':
                            save_dir = create_folders(label_str,dataset_name[j],flag, mask_ratio[j])
                        I=sgn_c_np[j,0]
                        Q = sgn_c_np[j, 1]
                        I2 = sgn_pred_np[j, 0]
                        Q2 = sgn_pred_np[j, 1]
                        loss_t=np.mean(np.abs(balence[i_loader] *sgn_c_np[j]-balence[i_loader] *sgn_pred_np[j]))
                        fig, axs = plt.subplots(2, 2, figsize=(8, 2.5))  # Set figsize to (6, 2)

                        # Plot in the first subplot
                        axs[0, 0].plot(t, I)
                        axs[0, 0].plot(t, Q)
                        axs[0, 0].set_title('sgn')

                        # Plot in the second subplot
                        axs[0, 1].plot(t, sgn_n_np[j, 0])
                        axs[0, 1].plot(t, sgn_n_np[j, 1])
                        axs[0, 1].set_title('sgn_noise')

                        # Plot in the third subplot
                        axs[1, 0].plot(t, I2)
                        axs[1, 0].plot(t, Q2)
                        axs[1, 0].set_title(f'Snr_{snrs[j]}_sgn_pred loss is {loss_t:.3f}')

                        # Adjust subplot spacing and margins
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6)

                        # Save the figure
                        filename = os.path.join(save_dir, f'{i}_{j}.jpg')  # Create unique filename
                        plt.savefig(filename, bbox_inches='tight', pad_inches=0,dpi=400)
                        plt.close()
                i=i+1
                # target_var = target_var.to(torch.float)

                loss1 = criterion(sgn_pred, sgn_c.to(device))
                loss2 = loss1
                loss=(1-args.lamba)*loss1+args.lamba*loss2
                # measure accuracy and record loss
                acc.update(0)
                # acc_AA(output, val_label,acc_AA_pre,acc_AA_count)
                losses_class.update(loss.item())
                losses_class1.update(loss1.item())
                losses_class2.update(loss2.item())


                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)
    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)
    return acc.avg, losses_class.avg


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch_seed,np_seed=set_random_seed(args.torch_seed, args.np_seed)
    norm_style='no'
    img_norm='maxmin'
    return_img=False
    # 读取YAML文件
    with open('scripts/multi_task_pretrain.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)

    data_paths = ['./data/RML2016_10a+b_gao17SNRdata.mat',
                  './data/RML2018.10a_train_data.npy',
                  './data/ADSB.pkl',
                  './data/WIFI_2ft.pkl'
                  ]
    Names =['RML2016','RML2018','ADSB','WIFI']

    split_ratios = [1.00, 1.00, 1.00, 0.05]
    val_split = 0.7
    Names, batch_sizes, num_workers, balence, thresholds, data_list, label_list = prepare_data(yaml_config,
                                                                                               split_ratios, Names,
                                                                                               data_paths)
    shapes = [data.shape for data in data_list if data is not None]
    print(*shapes, sep='\n')

    total_samples = sum(shape[0] for shape in shapes)
    probs = [shape[0] / total_samples for shape in shapes]
    print(probs[0])


    train_sets, train_loaders, val_loaders = [], [], []


    for data, label, name, batch_size, num_worker in zip(data_list, label_list, Names,
                                                         batch_sizes, num_workers):
        train_set = SigDataSet_freq_units2(data, label,
                                           newdata=args.newdata, adsbis=args.adsbis,
                                           resample_is=yaml_config[name]['resample_is'],
                                           samplenum=yaml_config[name]['resample_num'],
                                           resize_is=False,
                                           snr_range=[14, 40] if name != 'ADSB' else [16, 40],
                                           sgnaug=False, imgaug=False, sgn_expend=False,
                                           RGB_is=args.RGB_is, zhenshiSNR=False,
                                           sgn_norm=norm_style, img_norm=img_norm,
                                           return_img=return_img, freq_choose=args.trans_choose,
                                           window='None', Seed=np_seed,
                                           dataset_name=yaml_config[name]['dataset_name'])
        train_sets.append(train_set)
        train_set, val_set = split_dataset(valsplit=val_split, train_set=train_set)
        val_set.sgn_expend=False
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_worker, prefetch_factor=args.pref,
                                  drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_worker,
                                prefetch_factor=args.pref, shuffle=True, drop_last=True)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    task_name = 'pretrain'
    
    # model_name="Time_LLM_Units_ED_SIMCLR_hfpatch"
    if args.model_name=="RadioLLM":
        model = RadioLLM(args, yaml_config=yaml_config)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.cuda()

    load_dict(model, args.model_path, device)

    criterion =nn.MSELoss()

    criterion2 = nn.SmoothL1Loss(beta=0.05)
    criterion3 = nn.CrossEntropyLoss()
    criterion4 = nn.CrossEntropyLoss()

    num_training_steps = sum([len(loader) for loader in train_loaders]) * args.epochs
    warmup_steps = int(0.1 * num_training_steps)  # 设置warmup步数为总步数的10%

    optimizer_sgd = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.005,
        nesterov=True
    )

    optimizer_adam = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=5e-3
        )

    optimizer =  optimizer_sgd
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    csv_logger = CSVStats()
    early_stopping = EarlyStopping(save_path='./checkpoint_LLM/LLM/{}/{}/total/xiaorong/woHTRP/TOPK{}/{}/generate/{}'.
                                       format(args.model_name, args.llm_model,args.K,args.task_name,args.decoder_is), patience=args.patience,
                                       wait=args.wait, choose=args.trans_choose,save_best=True,is_LORA=args.is_LORA)
    wait_idem = args.wait
    declay_count = 0
    acc_train=0
    loss_train=0
    acc_val = 0
    loss_val = 0
    for epoch in range(0, args.epochs):

        acc_train, loss_train = trainhec(
            train_loaders, model, criterion,criterion2,criterion3,criterion4,
            optimizer, epoch, epoch_max=args.epochs,adsbis=args.adsbis,
            scheduler=scheduler,prob=probs,thresholds=thresholds,balence=balence,Names=Names)
        #
        # torch.cuda.empty_cache()

        acc_val, loss_val = validatehec(
            val_loaders, model, criterion, criterion2, criterion3, criterion4, epoch,
            epoch_max=args.epochs, adsbis=args.adsbis, scheduler=scheduler, prob=probs,
            thresholds=thresholds, balence=balence,Names=Names)
        scheduler.step()
        csv_logger.add(acc_train, acc_val, loss_train, loss_val, optimizer.param_groups[0]['lr'])
        csv_logger.write(patience=args.patience,wait=args.wait,choose=args.trans_choose,name=args.name,seed=0,few_shotnum=0)

        early_stopping(loss_val, model)
        if early_stopping.flag==True:
            wait_idem=args.wait
        if early_stopping.counter >5:
            # optimizer = optimizer_sgd
            wait_idem += 1
            if wait_idem>=args.wait:
                args.lr = adjust_learning_rate(optimizer, args.lr,args.declay)
                wait_idem=0
                declay_count+=1
            if  declay_count>=args.yuzhi:
                args.lr = adjust_learning_rate(optimizer, 0.001*(0.5)**3, args.declay)
                declay_count = 0
        print(args.wait)
        if early_stopping.early_stop:
            print("Early stopping")
            break
