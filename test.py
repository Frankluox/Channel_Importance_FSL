import argparse
from torchvision import transforms
from dataset import AircraftDataset, Chest, coco, general_dataset, ISIC, omniglot, OxfordFlowers102Dataset, CategoriesSampler
from architectures import get_backbone, get_classifier
import tqdm
import torch
import torch.nn.functional as F
from utils import count_acc, Averager
import numpy as np
import os
from torch.utils.data import DataLoader

backbones = ['resnet12', 'resnet50', 'WRN_28_10', 'conv-4', 'SEnet']

classifiers = ['proto_head', 'LR', "metaopt"]# LR for MoCo, S2M2; proto_head for PN, CE, Meta Baseline; metaopt for MetaOPT 

datasets = ['miniImageNet', 'CUB', 'Textures', 'Traffic_Signs',
            'Aircraft', 'Omniglot', 'VGG_Flower', 'MSCOCO', 'QuickDraw', 'Fungi', 
            'Plant_Disease', 'ISIC', 'EuroSAT', 'ChestX',
            'Real', 'Sketch', 'Infograph', 'Painting', 'Clipart']


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    # load pretrained model
    parser.add_argument('--backbone_name', type=str, default='resnet12', choices=backbones)
    parser.add_argument('--backbone_path', type=str, default=None, help='path to the pretrained backbone')
    parser.add_argument('--classifier_name', type=str, default='proto_head', choices=classifiers)

    # dataset
    parser.add_argument('--dataset_name', type=str, default='miniImageNet', choices=datasets)
    parser.add_argument('--dataset_root', type=str, default='', 
                        help='root directory of the dataset')
    


    # settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id if available')
    parser.add_argument('--num_test', type=int, default=5,
                        help='Number of test runs')
    parser.add_argument('--num_task', type=int, default=2000,
                        help='Number of tasks per run')
    parser.add_argument('--way', type=int, default=5,
                        help='Number of classes per task')
    parser.add_argument('--shot', type=int, default=5,
                        help='Number of support images per class')
    parser.add_argument('--num_query', type=int, default=15,
                        help='Number of query images per class')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='number of tasks per batch')

    opt = parser.parse_args()

    return opt

def main():
    args = parse_option()

   

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.backbone_name == 'resnet50':
        resize_sz = 256
        crop_sz = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        resize_sz = 92
        crop_sz = 84
        normalize = transforms.Normalize(mean=[0.4712, 0.4499, 0.4031],
                                     std=[0.2726, 0.2634, 0.2794])

    transform = transforms.Compose([
                            transforms.Resize([resize_sz, resize_sz]),
                            transforms.CenterCrop(crop_sz),
                            transforms.ToTensor(),
                            normalize])

    #obtain dataset
    if args.dataset_name == 'Aircraft':
        dataset = AircraftDataset(args.dataset_root, transform)
    elif args.dataset_name == 'Omniglot':
        dataset = omniglot(args.dataset_root, transform)
    elif args.dataset_name == 'VGG_Flower':
        dataset = OxfordFlowers102Dataset(args.dataset_root, transform)
    elif args.dataset_name == 'MSCOCO':
        dataset = coco(args.dataset_root, transform)
    elif args.dataset_name == 'ISIC':
        dataset = ISIC(args.dataset_root, transform)
    elif args.dataset_name == 'ChestX':
        dataset = Chest(args.dataset_root, transform)
    else:
        dataset = general_dataset(args.dataset_root, transform)

    # Logistic Regression passes single task
    if args.classifier_name == 'LR':
        args.batch_size = 1

    task_sampler = CategoriesSampler(dataset.label, args.num_task,
                                     args.way, args.shot+args.num_query, args.batch_size)

    data_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers,
            batch_sampler = task_sampler,
            pin_memory = True
        )
    
    
    model = get_backbone(args.backbone_name)
    state = torch.load(args.backbone_path)
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    classifier = get_classifier(args.classifier_name)
    

    
    label = torch.arange(args.way, dtype=torch.int8).repeat(args.num_query)
    label = label.type(torch.LongTensor).reshape(-1)
    label = torch.unsqueeze(label, 0).repeat(args.batch_size, 1).reshape(-1)
    if torch.cuda.is_available():
        label = label.cuda()

    #None: original performance. Simple: performance using simple transformation
    test_acc_record_None = np.zeros((args.num_test,))
    test_acc_record_Simple = np.zeros((args.num_test,))

    for i in range(1, args.num_test+1):
        print(f"The {i}-th test run:")
        ave_acc_None = Averager()
        ave_acc_Simple = Averager()
        data_loader_tqdm = tqdm.tqdm(data_loader)
        with torch.no_grad():
            for _, batch in enumerate(data_loader_tqdm, 1):
                data, _ = [_ for _ in batch]
                if torch.cuda.is_available():
                    data = data.cuda()
                num_support_samples = args.way * args.shot
                data = model(data)
                data = data.reshape([args.batch_size, -1] + list(data.shape[-3:]))
                data_support = data[:, :num_support_samples]
                data_query = data[:, num_support_samples:]

                logit_None = classifier(data_query, data_support, args.way, args.shot, False)
                logit_Simple = classifier(data_query, data_support, args.way, args.shot, True)
                logit_None = logit_None.reshape(label.size(0),-1)
                logit_Simple = logit_Simple.reshape(label.size(0),-1)

                acc_None = count_acc(logit_None, label) * 100
                acc_Simple = count_acc(logit_Simple, label) * 100
                ave_acc_None.add(acc_None)
                ave_acc_Simple.add(acc_Simple)


        test_acc_record_None[i-1] = ave_acc_None.item()
        test_acc_record_Simple[i-1] = ave_acc_Simple.item()
        print("The original accuracy for the {}-th test run: {:.2f}%".format(i,ave_acc_None.item()))
        print("The accuracy using simple transformation for the {}-th test run: {:.2f}%\n".format(i,ave_acc_Simple.item()))
    
    mean_None = np.mean(test_acc_record_None)
    confidence_interval_None = 1.96 * np.std(test_acc_record_None)
    mean_Simple = np.mean(test_acc_record_Simple)
    confidence_interval_Simple = 1.96 * np.std(test_acc_record_Simple)

    print("Average original accuracy with 95% confidence interval: {:.2f}% +- {:.2f}".format(mean_None, confidence_interval_None))
    print("Average accuracy using simple transformation with 95% confidence interval: {:.2f}% +- {:.2f}".format(mean_Simple, confidence_interval_Simple))
        
if __name__ == '__main__':
    main()
                
    

    
    

