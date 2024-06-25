import torch
from tqdm import tqdm
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.tools import same_seeds,get_project_path
import argparse
import os
from utils.eila import EILA

import torch.nn as nn 
# from torchattack import DIFGSM, MIFGSM, ad
import torch.nn.functional as F



def get_args():
    parser = argparse.ArgumentParser(description='EILA on ImageNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='the batch size when training')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size of the dataloader')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether use gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu_id')
    parser.add_argument('--num_worker', type=int, default=4)

    parser.add_argument('--save_dir', type=str, default='./result')

    # attack parameters
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='default momentum value')
    parser.add_argument('--resize_rate', type=float, default=0.9,
                        help='resize rate')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='diversity_prob')
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--min_value', type=float, default=0.0)


    args = parser.parse_args()
    return args



def main(args):

    device = torch.device(f'cuda:{args.gpu_id}')
    # dataset
    dataloader = get_dataset(args)
    # models
    models, metrix = get_models(args, device=device)
    ens_model =    [ 'resnet18', 'resnet50', 'inc_v3', 'swin_t', 'vit_t', 'deit_t']
    save_dir = os.path.join(args.save_dir, 'eila')
    label_dir = os.path.join(save_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)
    adv_dir = os.path.join(save_dir, 'adv')
    os.makedirs(adv_dir, exist_ok=True)
    clean_dir = os.path.join(save_dir, 'clean')
    os.makedirs(clean_dir, exist_ok=True)


    label_list = []
    for idx, (data, label) in enumerate(dataloader):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        attack_method = EILA(models=[models[i] for i in ens_model], device=device, eps=args.eps, alpha=args.alpha)
        adv_exp = attack_method(data, label)
        torch.save(adv_exp.detach().cpu(), os.path.join(adv_dir, 'batch_{}.pt'.format(idx)))
        torch.save(data.detach().cpu(), os.path.join(clean_dir, 'clean_{}.pt'.format(idx)))
        print('batch_{}.pt saved'.format(idx))
        label_list.append(label.cpu())

        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, n)
        

        if (idx+1) == 20:
            break
    torch.save(label_list, os.path.join(label_dir, 'label.pt'))

    # show result
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    avg_acc = 0
    num = 0
    for model_name, _ in models.items():
        
        print(f"|\t{model_name.ljust(10, ' ')}\t"
            f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
            f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
            f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
    print('-' * 73)



if __name__ == '__main__':
    args = get_args()
    same_seeds(args.seed)
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)
