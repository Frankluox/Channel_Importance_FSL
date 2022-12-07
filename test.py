import argparse
from torchvision import transforms
from dataset import AircraftDataset, Chest, coco, general_dataset, ISIC, omniglot, OxfordFlowers102Dataset, miniImageNet, CategoriesSampler
from architectures import get_backbone, get_classifier
import tqdm
import torch
import torch.nn.functional as F
from utils import count_acc, Averager
import numpy as np
import os
from torch.utils.data import DataLoader
import collections

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
    parser.add_argument('--statistics_root', type=str, default='./data_statistics', 
                        help='(for oracle only) root to saved dataset statistics')
    


    # settings
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
    parser.add_argument('--use_oracle', type=str, default="False",
                        help='whether use oracle transformation')

    opt = parser.parse_args()

    return opt



def compute_dataset_statistics(model, dataset_name, dataset, num_workers, statistics_root):
    dataloader = DataLoader(dataset, 128, shuffle=False, num_workers=num_workers, pin_memory=True)
    with torch.no_grad():
        class_features = collections.defaultdict(list)
        mean_ = []
        std_ = []
        num = []
        abs_mean = []
        print("calculating dataset statistics...")
        for i, (data, labels) in enumerate(tqdm.tqdm(dataloader)):
            batch_size = data.size(0)
            data = data.cuda()
            
            labels = labels.cuda()
            data = model(data)
            data = F.adaptive_avg_pool2d(data, 1).squeeze_(-1).squeeze_(-1)
            data = F.normalize(data, p=2, dim=1, eps=1e-12)
            for j in range(batch_size):
                class_features[int(labels[j])].append(data[j])

        for class_, features in class_features.items():
                features = torch.stack(features)
                features = F.normalize(features, p=2, dim=1, eps=1e-12)
                features_abs = torch.abs(features)
                num.append(features.size(0))
                mean_.append(torch.mean(features, dim=0))
                abs_mean.append(torch.mean(features_abs, dim=0))
                std_.append(torch.std(features, dim=0))
            
        mean_ = torch.stack(mean_).cpu().numpy()
        np.save(os.path.join(statistics_root, "meanof"+dataset_name+".npy"),mean_)
        abs_mean = torch.stack(abs_mean).cpu().numpy()
        np.save(os.path.join(statistics_root, "abs_meanof"+dataset_name+".npy"),abs_mean)
        std_ = torch.stack(std_).cpu().numpy()
        np.save(os.path.join(statistics_root, "stdof"+dataset_name+".npy"),std_)
        num = np.array(num)
        np.save(os.path.join(statistics_root, "numof"+dataset_name+".npy"),num)



def main():
    args = parse_option()
    args.use_oracle = False if args.use_oracle == "False" else True


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
    if args.dataset_name == 'miniImageNet':
        dataset = miniImageNet(args.dataset_root, transform)
    elif args.dataset_name == 'Aircraft':
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
    if args.classifier_name == 'LR' or args.use_oracle == True:
        args.batch_size = 1


    model = get_backbone(args.backbone_name)
    state = torch.load(args.backbone_path)
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    
    
    if args.use_oracle == True:
        args.way = 2# Oracle transformation is used under binary tasks
        if not os.path.exists(args.statistics_root):
            os.makedirs(args.statistics_root)
        if not os.path.exists(os.path.join(args.statistics_root,"meanof"+args.dataset_name+".npy")):
            compute_dataset_statistics(model, args.dataset_name, dataset, args.num_workers, args.statistics_root)
    
    if args.use_oracle:
        classifier = get_classifier(
                        args.classifier_name, 
                        use_Oracle=True, 
                        statistics_root=args.statistics_root, 
                        dataset_name=args.dataset_name)
    else:
        classifier = get_classifier(args.classifier_name, use_Oracle=False)
    

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
    
    
    
    

    
    query_label = torch.arange(args.way, dtype=torch.int8).repeat(args.num_query)
    query_label = query_label.type(torch.LongTensor).reshape(-1)
    query_label = torch.unsqueeze(query_label, 0).repeat(args.batch_size, 1).reshape(-1)
    if torch.cuda.is_available():
        query_label = query_label.cuda()

    #None: original performance. Simple: performance using simple transformation
    test_acc_record_None = np.zeros((args.num_test,))
    test_acc_record_Simple = np.zeros((args.num_test,))

    if args.use_oracle == True:
        test_acc_record_Oracle = np.zeros((args.num_test,))

    for i in range(1, args.num_test+1):
        print(f"The {i}-th test run:")
        ave_acc_None = Averager()
        ave_acc_Simple = Averager()
        if args.use_oracle == True:
            ave_acc_Oracle = Averager()
        data_loader_tqdm = tqdm.tqdm(data_loader)
        with torch.no_grad():
            for _, batch in enumerate(data_loader_tqdm, 1):
                data, labels = [_ for _ in batch]
                if args.use_oracle == True:
                    all_labels = []
                    for label in labels:
                        j = int(label)
                        if j not in all_labels:
                            all_labels.append(j)
                if torch.cuda.is_available():
                    data = data.cuda()
                num_support_samples = args.way * args.shot
                data = model(data)
                data = data.reshape([args.batch_size, -1] + list(data.shape[-3:]))
                data_support = data[:, :num_support_samples]
                data_query = data[:, num_support_samples:]

                logit_None = classifier(data_query, data_support, args.way, args.shot, False, False)
                logit_Simple = classifier(data_query, data_support, args.way, args.shot, True, False)

                if args.use_oracle == True:
                    logit_Oracle = classifier(data_query, data_support, args.way, args.shot, False, True, all_labels)
                    # print(logit_Oracle.shape)
                    logit_Oracle = logit_Oracle.reshape(query_label.size(0),-1)
                    acc_Oracle = count_acc(logit_Oracle, query_label) * 100
                    ave_acc_Oracle.add(acc_Oracle)

                logit_None = logit_None.reshape(query_label.size(0),-1)
                logit_Simple = logit_Simple.reshape(query_label.size(0),-1)

                acc_None = count_acc(logit_None, query_label) * 100
                acc_Simple = count_acc(logit_Simple, query_label) * 100
                ave_acc_None.add(acc_None)
                ave_acc_Simple.add(acc_Simple)


        test_acc_record_None[i-1] = ave_acc_None.item()
        test_acc_record_Simple[i-1] = ave_acc_Simple.item()
        if args.use_oracle == True:
            test_acc_record_Oracle[i-1] = ave_acc_Oracle.item()
        print("The original accuracy for the {}-th test run: {:.2f}%".format(i,ave_acc_None.item()))
        print("The accuracy using simple transformation for the {}-th test run: {:.2f}%\n".format(i,ave_acc_Simple.item()))
        if args.use_oracle == True:
            print("The accuracy using oracle transformation for the {}-th test run: {:.2f}%\n".format(i,ave_acc_Oracle.item()))
    
    mean_None = np.mean(test_acc_record_None)
    confidence_interval_None = 1.96 * np.std(test_acc_record_None)
    mean_Simple = np.mean(test_acc_record_Simple)
    confidence_interval_Simple = 1.96 * np.std(test_acc_record_Simple)
    if args.use_oracle == True:
        mean_Oracle = np.mean(test_acc_record_Oracle)
        confidence_interval_Oracle = 1.96 * np.std(test_acc_record_Oracle)

    print("Average original accuracy with 95% confidence interval: {:.2f}% +- {:.2f}".format(mean_None, confidence_interval_None))
    print("Average accuracy using simple transformation with 95% confidence interval: {:.2f}% +- {:.2f}".format(mean_Simple, confidence_interval_Simple))
    if args.use_oracle == True:
        print("Average accuracy using oracle transformation with 95% confidence interval: {:.2f}% +- {:.2f}".format(mean_Oracle, confidence_interval_Oracle))
        
if __name__ == '__main__':
    main()
                
    

    
    

