import os
import timm
import yaml
import torch
from torchvision import transforms
from utils.AverageMeter import AccuracyMeter

yaml_path = '../configs/checkpoint.yaml'
test_path = '../configs/test.yaml'



def get_models(args, device):
    metrix = {}
    with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'imagenet':
        print (timm.list_models())
        for key, value in yaml_data.items():
            # print (value)
            model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            model.eval()

            if 'inc' in key or 'vit' in key or 'bit' in key or 'mixer' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  model)

            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    return models, metrix


def get_test_models(args, device):
    metrix = {}
    with open(os.path.join(args.root_path, 'utils', test_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'imagenet':

        for key, value in yaml_data.items():
            model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            model.eval()

            if 'inc' in key or 'vit' in key in key or 'mixer' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  model)
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')

    return models, metrix

