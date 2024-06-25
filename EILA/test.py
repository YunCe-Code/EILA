import torch
from tqdm import tqdm
# from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from utils.get_models import get_models, get_test_models
from utils.tools import same_seeds, get_project_path
import argparse
import os
import tensorflow as tf 


def get_args():
    parser = argparse.ArgumentParser(description='Test transferability of ImageNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size of the dataloader')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether use gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu_id')
    parser.add_argument('--num_worker', type=int, default=4)

    parser.add_argument('--adv_path', type=str, default='./result/eila/adv')
    parser.add_argument('--clean_path', type=str, default='./result/eila/clean')
    parser.add_argument('--label_path', type=str, default='./result/eila/labels')
    # attack parameters
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    models, metrix = get_test_models(args, device=device)
    labels = torch.load(os.path.join(args.label_path, "label.pt"))
    for i, _ in enumerate(tqdm(labels)):
        # print (i)
        x_adv = torch.load(os.path.join(args.adv_path, "batch_{}.pt".format(i)))
        x = torch.load(os.path.join(args.clean_path, "clean_{}.pt".format(i)))
        x, x_adv, label = x.to(device), x_adv.to(device), labels[i].to(device)
        print (torch.max(x_adv-x))
        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(x)
                r_adv = model(x_adv)

            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, label.size(0))

            
        

    # show result
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    cnns = ['wrn101_2', 'resnet18', 'resnet50', 'inc_v3', 'bit50_1', 'bit101_1', 
            'densenet121', 'resnet152']
    vits = ['vit_b', 'vit_t', 'deit_b', 'deit_t', 'swin_b', 'swin_s']
    avg_rate = 0
    num = 0

    for model_name, _ in models.items():
        avg_rate += round(metrix[model_name].attack_rate * 100,2)
        num+=1
        
        print(f"|\t{model_name.ljust(10, ' ')}\t"
            f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
            f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
            f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
    print('-' * 73)
    print ('avg_rate:{}'.format(avg_rate/num))
    # print ('CNNs average rate:{}'.format(avg_cnns/cnns_num))
    # print ('ViTs average rate:{}'.format(avg_vits/vits_num))


if __name__ == '__main__':
    args = get_args()
    same_seeds(args.seed)
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)
