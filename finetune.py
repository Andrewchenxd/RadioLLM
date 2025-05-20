import time
import matplotlib.pyplot as plt
import torch.optim
from torch.utils.data import DataLoader
from torch import nn
from dataset import *
from tqdm import tqdm
from utils.utils import *
from model.radiollm import RadioLLM
import argparse
import warnings
import copy
import torch.nn.functional as F
import yaml
import pickle
from sklearn.metrics import confusion_matrix, cohen_kappa_score
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast, GradScaler
choose = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


if choose==True:
    parser = argparse.ArgumentParser(description='Train TransNet')
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5) #5e-5
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument('--numclass', type=int, default=11, help='num of heads')
    parser.add_argument("--pretrain",type=bool,default=False)
    parser.add_argument("--lamba", default=1) #sgn clean
    parser.add_argument("--alpha", default=1) #sgn mask
    parser.add_argument("--beta", default=0.0)  # fliter
    parser.add_argument("--balence", default=1e3)
    parser.add_argument("--autocast", default=False)
    parser.add_argument("--softloss", default=False)
    parser.add_argument("--swav_is", type=str2bool, default=False)
    parser.add_argument("--decoder_is", type=str2bool, default=True)
    parser.add_argument("--norm_style", type=str, default="maxmin-1")
    # parser.add_argument("--model_path", type=str, default= './checkpoint_SwinTransformer_RML2016.10b_conmaehec/mask_ratio=0.75_suiji/patch_size_4_window_size_2/pwvd_best_network_loss_0.1398216283469686.pth')
    parser.add_argument("--model_path", type=str, default=
    './checkpoint_LLM/LLM/Time_LLM_Units_ED_SIMCLR_hfpatch/GPT2/total/soft_hard_prompt3/generate/best_network.pth')
    parser.add_argument("--acc_it", type=int, default=32)
    parser.add_argument("--clip_grad", type=int, default=100)
    parser.add_argument("--few_shotnum",type=int,default=1) # 微调数目

    parser.add_argument('--task_name', type=str, required=False, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

    parser.add_argument("--simclr_ratio", type=float, default=1, help="dropout")
    parser.add_argument("--soft_ratio", type=float, default=0.0, help="dropout")
    parser.add_argument("--dist_type", type=str, default="euc", help="distance type: cos, euc")
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
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--llm_path', type=str, default='./pretrain_model/gpt2',
                        help='LLM_model_path')

    parser.add_argument('--llm_dim', type=int, default='768',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--right_prob', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=7, help='top K prompt')
    parser.add_argument('--is_LORA', type=str2bool, default=True)
    parser.add_argument("--load_lora_only",type=bool,default=True)
    parser.add_argument("--decode_mask", type=str2bool, default=True)
    parser.add_argument("--mix", type=str2bool, default=True)
    parser.add_argument("--attn", type=str, default='prob')
    parser.add_argument('--d_ff2', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--nmb_prototypes', type=int, default=30, help='nmb_prototypes')
    parser.add_argument('--T', type=float, default=0.07, help='tempeture')
    parser.add_argument('--epsilon', type=float, default=0.05, help=' ')
    parser.add_argument('--sinkhorn_iterations', type=int, default=3, help=' ')

    parser.add_argument("--RGB_is", type=str2bool, default=True)
    parser.add_argument("--adsbis", type=str2bool, default=False)
    parser.add_argument("--resample", type=str2bool, default=False)
    parser.add_argument("--chazhi", type=str2bool, default=False)
    parser.add_argument("--newdata", type=str2bool, default=False)
    parser.add_argument("--numworks", type=int, default=4)
    parser.add_argument("--trans_choose", type=str, default='pwvd')
    parser.add_argument("--dataset", type=str, default='rml16a')
    parser.add_argument("--name", type=str, default='rml16a')
    parser.add_argument("--cnum", type=int, default=2)
    parser.add_argument("--samplenum", type=int, default=1)  # samplenum pwvd 15 without 5
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--wait", type=int, default=5)
    parser.add_argument("--declay", default=0.5)
    parser.add_argument("--yuzhi", type=int, default=10)
    parser.add_argument("--pref", type=int, default=20)

    parser.add_argument("--torch_seed", type=str, default='42')
    parser.add_argument("--np_seed", type=str, default='None')
    opt = parser.parse_args()


def acc_classes(pre, labels, BATCH_SIZE):
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc


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


def acc_snrs(pre, labels, snr, acc_snr_pre, acc_snr_count):
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_snr_pre[0, snr[i]] += 1
            acc_snr_count[0, snr[i]] += 1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre, acc_snr_count


def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
    return x


def calculate_accuracy(true_label, pre_label, classesnum):
    # 计算混淆矩阵
    cm = confusion_matrix(true_label, pre_label)

    # 初始化准确率矩阵
    accuracy_matrix = np.zeros((1, classesnum))

    # 计算每个类别的准确率
    for i in range(classesnum):
        accuracy_matrix[0, i] = cm[i, i] / np.sum(cm[i, :])

    # 计算总体准确率（Overall Accuracy，OA）
    OA = np.trace(cm) / np.sum(cm)

    # 计算平均准确率（Average Accuracy，AA）
    AA = np.mean(accuracy_matrix)
    accuracy_matrix = accuracy_matrix.flatten()
    return accuracy_matrix, OA, AA


def train(train_loader, model, criterion, optimizer, epoch, epoch_max, batchsize, model_name, adsbis=False):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre = np.zeros((1, 31))
        acc_snr_count = np.zeros((1, 31))
    elif args.trans_choose == "pwvd" or args.trans_choose == "gasf":
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    elif args.trans_choose == "pwvd_2018":
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    # switch to train mode
    scaler = GradScaler()
    model.train()

    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input1, target, snr) in enumerate(train_loader):
            images, labels = input1, target
            with autocast(enabled=args.autocast):
                if args.swav_is==True:
                    if args.softloss == False:
                        output = model(images.to(device),images.to(device), enable_mask=False, dataset_name='RML2016a+b_total_snr')
                    else:
                        output = model(images.to(device), images.to(device), images.to(device), enable_mask=False,
                                       dataset_name='RML2016a+b_total_snr')
                else:
                    output = model(images.to(device), enable_mask=False,
                                   dataset_name='RML2016a+b_total_snr')

                target_var = labels.to(device)
                # target_var = target_var.to(torch.float)
                loss = criterion(output, target_var)

            # measure accuracy and record loss
            acc.update(acc_classes(output.data, target, batchsize))
            if adsbis == True:
                acc_snrs(output, labels, snr - 10, acc_snr_pre, acc_snr_count)
            else:
                acc_snrs(output, labels, snr, acc_snr_pre, acc_snr_count)
            losses_class.update(loss.item())

            # compute gradient and do SGD step
            if args.autocast:
                optimizer.zero_grad()
                scaler.scale(loss.mean()).backward()
                # 使用 scaler 调用 optimizer.step()
                scaler.step(optimizer)
                # 更新 scaler 状态
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_postfix(**{'train_loss_': losses_class.avg, 'acc': acc.avg})
            pbar.update(1)
    print(acc_snr_pre / acc_snr_count * 100)

    return acc.avg, losses_class.avg

def validate(val_loader, model, criterion, epoch, epoch_max, batchsize, adsbis=False):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 31))
        acc_snr_count_val = np.zeros((1, 31))
    elif args.trans_choose == "pwvd" or args.trans_choose == "gasf":
        acc_snr_pre_val = np.zeros((1, 21))
        acc_snr_count_val = np.zeros((1, 21))
    elif args.trans_choose == "pwvd_2018":
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    # switch to evaluate mode
    model.eval()
    scaler = GradScaler()

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1, target, snr) in enumerate(val_loader):
                val_image, val_label = input1, target
                with autocast(enabled=args.autocast):
                    if args.swav_is == True:
                        if args.softloss == False:
                            output = model(val_image.to(device), val_image.to(device), enable_mask=False,
                                           dataset_name='RML2016a+b_total_snr')
                        else:
                            output = model(val_image.to(device), val_image.to(device), val_image.to(device), enable_mask=False,
                                           dataset_name='RML2016a+b_total_snr')
                    else:
                        output = model(val_image.to(device), enable_mask=False,
                                       dataset_name='RML2016a+b_total_snr')

                    target_var = val_label.to(device)
                    # target_var = target_var.to(torch.float)
                    loss = criterion(output, target_var)

                # measure accuracy and record loss
                acc.update(acc_classes(output.data, target, batchsize))
                if adsbis == True:
                    acc_snrs(output, val_label, snr - 10, acc_snr_pre_val, acc_snr_count_val)
                else:
                    acc_snrs(output, val_label, snr, acc_snr_pre_val, acc_snr_count_val)
                # acc_AA(output, val_label,acc_AA_pre,acc_AA_count)
                losses_class.update(loss.item())

                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)
    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    return acc.avg, losses_class.avg

def validate_forresult_all(val_loader, model, criterion, epoch, epoch_max, batchsize, model_name, adsbis=False):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre = np.zeros((1, 31))
        acc_snr_count = np.zeros((1, 31))
    elif args.trans_choose == "pwvd" or args.trans_choose == "gasf":
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    elif args.trans_choose == "pwvd_2018":
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    # switch to evaluate mode
    model.eval()
    true_label = []
    pre_label = []
    SNR = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1, target, snr) in enumerate(val_loader):
                val_image, val_label = input1, target

                if args.swav_is == True:
                    if args.softloss == False:
                        output = model(val_image.to(device), val_image.to(device), enable_mask=False,
                                       dataset_name='RML2016a+b_total_snr')
                    else:
                        output = model(val_image.to(device), val_image.to(device), val_image.to(device),
                                       enable_mask=False,
                                       dataset_name='RML2016a+b_total_snr')
                else:
                    output = model(val_image.to(device), enable_mask=False,
                                   dataset_name='RML2016a+b_total_snr')

                pre_y = torch.max(output, dim=1)[1]
                pre_y = pre_y.detach().cpu().numpy()
                snr = snr.detach().cpu().numpy()
                val_label = val_label.detach().cpu().numpy()
                SNR.append(snr)
                pre_label.append(pre_y)
                true_label.append(val_label)
                # target_var = target_var.to(torch.float)
                loss = 0
                if adsbis == True:
                    acc_snrs(output, val_label, snr - 10, acc_snr_pre, acc_snr_count)
                else:
                    acc_snrs(output, val_label, snr, acc_snr_pre, acc_snr_count)
                # measure accuracy and record loss
                acc.update(acc_classes(output.data, target, batchsize))

                # acc_AA(output, val_label,acc_AA_pre,acc_AA_count)
                losses_class.update(loss)

                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)
    true_label = np.concatenate(true_label)
    pre_label = np.concatenate(pre_label)
    SNR = np.concatenate(SNR)
    print(acc_snr_pre / acc_snr_count * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    return acc.avg, losses_class.avg, true_label, pre_label, SNR


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = parser.parse_args()
    print(args.numclass)
    model_name = 'RadioLLM'

    with open('scripts/multi_task_pretrain.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)
    if args.trans_choose == "pwvd" or args.trans_choose == "stft":
        norm_style = args.norm_style
        print('trans_choose:{}'.format(args.trans_choose))
        print('norm_style:{}'.format(norm_style))
        print('dataset:{}'.format(args.dataset))
        train_data_path = 'data/{}/train'.format(args.name)
        val_data_path = 'data/{}/val'.format(args.name)
        test_data_path = 'data/{}/test'.format(args.name)

        train_set = SigDataSet_sgn_npy(train_data_path,few_shotnum=args.few_shotnum, newdata=args.newdata,
                                           adsbis=args.adsbis,
                                           resample_is=args.resample, samplenum=args.samplenum, resize_is=False,
                                           norm=norm_style,
                                           snr_range=[0, 20], sgnaug=True, return_label=True,pretrain=args.pretrain)
        val_set = SigDataSet_sgn_npy(val_data_path, few_shotnum=0, newdata=args.newdata,
                                         adsbis=args.adsbis,
                                         resample_is=args.resample, samplenum=args.samplenum, resize_is=False,
                                         norm=norm_style,
                                         snr_range=[0, 20], sgnaug=False, return_label=True)
        
        test_set = SigDataSet_sgn_npy(test_data_path, few_shotnum=0, newdata=args.newdata,
                                         adsbis=args.adsbis,
                                         resample_is=args.resample, samplenum=args.samplenum, resize_is=False,
                                         norm=norm_style,
                                         snr_range=[0, 20], sgnaug=False, return_label=True)

    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.numworks, prefetch_factor=args.pref, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=args.numworks \
                            , prefetch_factor=args.pref, shuffle=False, pin_memory=True, persistent_workers=True )
    test_loader = DataLoader(test_set, batch_size=args.batchsize, num_workers=args.numworks
                            , prefetch_factor=args.pref, shuffle=False, pin_memory=True, persistent_workers=True )
    if model_name=='Time_LLM_Units':
        model=Time_LLM_Units_ED_hfpatch(args,yaml_config)


    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.cuda()
        print("DataParallel devices:", model.device_ids)  # 输出设备 ID 列表
    else:
        model = model.cuda()
        
    # try:
    #     model_path=f"./checkpoint_Spec2sgn_{args.dataset}/fintune/seed_42_fewshot_{args.few_shotnum}/Time_LLM_Units/best_network.pth"
    #     load_dict(model, model_path, device)
    # except:
    load_dict(model, args.model_path, device)

    criterion = nn.CrossEntropyLoss()
    beta = 0.75

    optimizer_sgd = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.005,
        nesterov=True
    )
    if args.dataset == 'ADSB':
        # Set different learning rates for MLP layers and other layers
        mlp_params = [p for n, p in model.named_parameters() if 'mlp' in n]
        other_params = [p for n, p in model.named_parameters() if 'mlp' not in n]
        
        optimizer_adam = torch.optim.AdamW([
            {'params': mlp_params, 'lr': args.lr * 5},  # MLP layers with a lower learning rate
            {'params': other_params, 'lr': args.lr}       # Other layers with the default learning rate
        ],
        betas=(0.9, 0.95),
        weight_decay=0.3)
    else:
        optimizer_adam = torch.optim.AdamW(model.parameters(),
                                           lr=args.lr,
                                           betas=(0.9, 0.95),
                                           weight_decay=0.3)
    optimizer = optimizer_adam
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-8)
    csv_logger = CSVStats(save_dir='./runs/{}/{}/{}/{}'.format(args.task_name,model_name,args.llm_model,args.name))
    early_stopping = EarlyStopping(save_path='./checkpoint_Spec2sgn_{}/fintune/seed_{}_fewshot_{}/{}/'.
                                   format(args.dataset,args.torch_seed,args.few_shotnum,model_name), patience=args.patience,
                                   wait=args.wait, choose=args.trans_choose,is_LORA=args.is_LORA ,save_best=True
                                   )
    wait_idem = args.wait
    declay_count = 0
    acc_train = 0
    loss_train = 0
    save_dir='./result/{}/{}/{}/{}'.format(args.task_name,model_name,args.llm_model,args.name)
    os.makedirs(save_dir,exist_ok=True)
    best_val=0
    start_index=0
    for epoch in range(start_index, args.epochs):
        if epoch==start_index:
            print("TRAINING_START", flush=True)
        acc_train, loss_train = train(train_loader, model, criterion, optimizer, epoch, batchsize=args.batchsize, epoch_max=args.epochs, adsbis=args.adsbis, model_name=model_name)
        
        acc_val, loss_val, true_label, pre_label, SNR = validate_forresult_all(val_loader, model, criterion, epoch, epoch_max=args.epochs, batchsize=args.batchsize, adsbis=False, model_name=model_name)
    
        if acc_val>best_val:
            best_val=acc_val
            result = {'true_label': true_label, 'pre_label': pre_label, 'SNR': SNR}
            save_path = os.path.join(save_dir, 'best_result_fewshot{}_seed{}_epoch{}.mat'.format(args.torch_seed, args.few_shotnum, epoch))
            scio.savemat(save_path, result)

        csv_logger.add(acc_train, acc_val, loss_train, loss_val, args.lr)
        csv_logger.write(patience=args.patience, wait=args.wait, choose=args.trans_choose, name=args.name, seed=args.torch_seed, few_shotnum=args.few_shotnum)

        # lr_scheduler.step()
        early_stopping(loss_val, model)
        if early_stopping.counter >= args.wait:
            args.lr = adjust_learning_rate(optimizer, args.lr, args.declay)
        if early_stopping.early_stop == True:
            print("Early stopping")
            break
    print("TRAINING_END", flush=True)
    acc_test, loss_test, true_label, pre_label, SNR = validate_forresult_all(test_loader, model, criterion, epoch, epoch_max=args.epochs, batchsize=args.batchsize, adsbis=False, model_name=model_name)
    result = {'true_label': true_label, 'pre_label': pre_label, 'SNR': SNR}
    save_path = os.path.join(save_dir, 'test_best_result_fewshot{}_seed{}_epoch{}.mat'.format(args.torch_seed, args.few_shotnum, epoch))
    scio.savemat(save_path, result)