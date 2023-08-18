from numpy.core.records import array
from numpy.lib.function_base import quantile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model, load_model_ex
from evaluate import mean_average_precision
from torch.nn import Parameter
from wta import WTA
import statistics
import pickle

def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          train_dataloader_wag,
          multi_labels,
          code_length,
          num_features,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          threshold,
          eta_1,
          eta_2,
          eta_3,
          evaluate_interval,
          number_permutation,
          K,
          percentile,
          eta_4,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """
    # Model, optimizer, criterion

    

    model = load_model(arch, code_length)
    model.to(device)
    model_ex = load_model_ex()
    model_ex.to(device)

    base_params = list(map(id, model.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": lr},
        {"params": model.model.parameters(), "lr": lr * 0.01},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-5)


    # Extract features
    for i in range(1):

        if eta_4== 10:
            features, _ = extract_features(model, train_dataloader_wag, num_features, device, verbose)
            logger.info('Get the deep feature')

            S2 = generate_wta_based_similarity(features, eta_3, number_permutation, K)
        elif eta_4>=0 and eta_4<=7:
            S2 = generate_wta_based_similarity2(eta_3, number_permutation, K, eta_4)
        elif eta_4==100:
            S2 = generate_wta_based_similarity3(eta_3, number_permutation, K)

        if eta_3==0:
            S2 = torch.load('S2_flick_old_{}_{}.pt'.format(number_permutation, K))
        elif eta_3==2:
            S2 = torch.load('S2_cifar_{}_{}.pt'.format(number_permutation, K))
        elif eta_3==3:
            S2 = torch.load('S2_nus21_{}_{}.pt'.format(number_permutation, K))
        elif eta_3==4:
            S2 = torch.load('S2_coco_{}_{}.pt'.format(number_permutation, K))
        threshold_s = torch.kthvalue(S2.flatten(), int(S2.numel() * (1 - percentile))).values
        S = (S2 < threshold_s) * 0.0 + (S2 >= threshold_s) * 1.0

        S2 = S2/512.0
        S = S.to(device)
        S2 = S2.to(device)

        mask = torch.eye(S2.size(0), dtype=torch.bool, device=S2.device)

        max_value = S2[~mask].max()
        min_value = S2[~mask].min()

        pos = max_value - threshold_s/512.0
        neg = threshold_s/512.0 - min_value
        logger.info('END')
        

        # Training
        model.train()
        model_ex.eval()
        for epoch in range(max_iter):
            for i, (data, data_aug,data2,_, index) in enumerate(train_dataloader):


                data = data.to(device)
                batch_size = data.shape[0]
                data_aug = data_aug.to(device)
                data2 = data2.to(device)


                optimizer.zero_grad()

                f_1, v= model(data)
                f_2, v_aug= model(data_aug)
                _,v_quan = model(data2)

                h_quan = F.normalize(v_quan)



                targets = S[index, :][:, index]
                weights = S2[index, :][:, index]



                theta_exp = torch.exp(h_quan.mm(h_quan.t()) / 0.1)
                the_frac = ((1 - targets) * ((weights - threshold_s / 512.0) / (-neg)) * theta_exp).sum(1).view(-1,1) + 0.00001
                wta_contrast_loss = - (torch.log(theta_exp / the_frac) * targets * ((weights - threshold_s / 512.0) / pos)).sum() / targets.sum()




                c = F.normalize(v,dim=0).T @ F.normalize(v_aug,dim=0)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
                off_diag = off_diagonal(c).pow(2).mean()

                bt_loss = on_diag + off_diag

                quan_loss = - torch.mean(torch.sum(F.normalize(h_quan,dim=-1)*F.normalize(torch.sign(h_quan),dim=-1), dim=1))

                loss = eta_2 * quan_loss + bt_loss * eta_1  + wta_contrast_loss


                loss.backward()
                optimizer.step()


                # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1) or (epoch==0):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                multi_labels,
                                )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    #np.save("./code_nus21/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())

    #np.save("./code_nus21/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())

    #np.save("./code_nus21/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())

    #np.save("./code_nus21/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            _, outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code



def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    features_2 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, data_2 ,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            features_1[index, :] = model(data_1).cpu()
            features_2[index, :] = model(data_2).cpu()

    model.set_extract_features(False)
    model.train()
    return features_1, features_2



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    




def generate_wta_based_similarity(features, eta_3, number_permutation, K):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        threshold: Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """


    features = features.cpu().numpy()
    
    wta_model = WTA(
        K=K,
        number_of_permutations=1
    )

    
    new_features_c = np.zeros((features.shape[0], number_permutation))
    for i in range(number_permutation):
        theta = np.random.permutation(features.shape[1])
        new_features_one = wta_model.hash(features, theta)
        for j in range(features.shape[0]):
            new_features_c [j,i] = new_features_one[j].index(max(new_features_one[j]))

    if eta_3==2:
        np.save('cifar_{}_{}.npy'.format(number_permutation, K), new_features_c)
    elif eta_3==3:
        np.save('nus21_{}_{}.npy'.format(number_permutation, K), new_features_c)
    elif eta_3==0:
        np.save('flick_old_{}_{}.npy'.format(number_permutation, K), new_features_c)
    elif eta_3==4:
        np.save('new_coco_{}_{}.npy'.format(number_permutation, K), new_features_c)

    sys.exit(0)



def generate_wta_based_similarity2(eta_3, number_permutation, K, eta_4):

    new_features_c = np.load('cifar_{}_{}.npy'.format(number_permutation, K))
    number_training_samples = new_features_c.shape[0]

    new_features_c_tensor = torch.from_numpy(new_features_c)
    new_features_c_tensor1 = new_features_c_tensor.unsqueeze(1)
    new_features_c_tensor1 = new_features_c_tensor1.expand(number_training_samples, number_training_samples, new_features_c.shape[1])

    if eta_3 == 2:
        if eta_4 == 0:
            S = new_features_c_tensor1[0:625, :, :] - new_features_c_tensor
            S1 = torch.count_nonzero(S, dim=2)
            S2_ = number_permutation - S1
            np.save('cifar_{}_{}_{}_0.npy'.format(number_permutation, K, eta_4), S2_)
        elif eta_4 == 1:
            Ss = new_features_c_tensor1[625:1250, :, :] - new_features_c_tensor
            S1s = torch.count_nonzero(Ss, dim=2)
            S2_s = number_permutation - S1s
            np.save('cifar_{}_{}_{}_1.npy'.format(number_permutation, K, eta_4 - 1), S2_s)
        if eta_4 == 2:
            S = new_features_c_tensor1[1250:1875, :, :] - new_features_c_tensor
            S1 = torch.count_nonzero(S, dim=2)
            S2_ = number_permutation - S1
            np.save('cifar_{}_{}_{}_2.npy'.format(number_permutation, K, eta_4-2), S2_)
        elif eta_4 == 3:
            Ss = new_features_c_tensor1[1875:2500, :, :] - new_features_c_tensor
            S1s = torch.count_nonzero(Ss, dim=2)
            S2_s = number_permutation - S1s
            np.save('cifar_{}_{}_{}_3.npy'.format(number_permutation, K, eta_4 - 3), S2_s)
        elif eta_4 == 4:
            S_1 = new_features_c_tensor1[2500:3125, :, :] - new_features_c_tensor
            S1_1 = torch.count_nonzero(S_1, dim=2)
            S2_1 = number_permutation - S1_1
            np.save('cifar_{}_{}_{}_0.npy'.format(number_permutation, K, eta_4 - 3), S2_1)
        elif eta_4 == 5:
            S_1s = new_features_c_tensor1[3125:3750, :, :] - new_features_c_tensor
            S1_1s = torch.count_nonzero(S_1s, dim=2)
            S2_1s = number_permutation - S1_1s
            np.save('cifar_{}_{}_{}_1.npy'.format(number_permutation, K, eta_4 - 4), S2_1s)
        elif eta_4 == 6:
            S_1 = new_features_c_tensor1[3750:4375, :, :] - new_features_c_tensor
            S1_1 = torch.count_nonzero(S_1, dim=2)
            S2_1 = number_permutation - S1_1
            np.save('cifar_{}_{}_{}_2.npy'.format(number_permutation, K, eta_4 - 5), S2_1)
        elif eta_4 == 7:
            S_1s = new_features_c_tensor1[4375:, :, :] - new_features_c_tensor
            S1_1s = torch.count_nonzero(S_1s, dim=2)
            S2_1s = number_permutation - S1_1s
            np.save('cifar_{}_{}_{}_3.npy'.format(number_permutation, K, eta_4 - 6), S2_1s)

        sys.exit(0)

    return S2

def generate_wta_based_similarity3(eta_3, number_permutation, K):
    if eta_3 == 2:
        S2_ = torch.FloatTensor(np.load('cifar_{}_{}_0_0.npy'.format(number_permutation, K)))
        S2_s = torch.FloatTensor(np.load('cifar_{}_{}_0_1.npy'.format(number_permutation, K)))
        S2_s2 = torch.FloatTensor(np.load('cifar_{}_{}_0_2.npy'.format(number_permutation, K)))
        S2_s3 = torch.FloatTensor(np.load('cifar_{}_{}_0_3.npy'.format(number_permutation, K)))
        S2_1 = torch.FloatTensor(np.load('cifar_{}_{}_1_0.npy'.format(number_permutation, K)))
        S2_1s = torch.FloatTensor(np.load('cifar_{}_{}_1_1.npy'.format(number_permutation, K)))
        S2_1s2 = torch.FloatTensor(np.load('cifar_{}_{}_1_2.npy'.format(number_permutation, K)))
        S2_1s3 = torch.FloatTensor(np.load('cifar_{}_{}_1_3.npy'.format(number_permutation, K)))

        S2_0 = torch.cat((S2_, S2_s))
        S2_02 = torch.cat((S2_0, S2_s2))
        S2_03 = torch.cat((S2_02, S2_s3))
        S2_0s = torch.cat((S2_03, S2_1))
        S2_0s1 = torch.cat((S2_0s, S2_1s))
        S2_0s2 = torch.cat((S2_0s1, S2_1s2))
        S2 = torch.cat((S2_0s2, S2_1s3))
        torch.save(S2, 'S2_cifar_{}_{}.pt'.format(number_permutation, K))
        return S2