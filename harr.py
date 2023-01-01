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
from model_loader import load_model
from evaluate import mean_average_precision
from torch.nn import Parameter
from losses import SupConLoss
from wta import WTA

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
          evaluate_interval,
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
    base_params = list(map(id, model.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": lr},
        {"params": model.model.parameters(), "lr": lr * 0.01},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-5)
    criterion1 = SupConLoss(temperature=0.07).cuda()

    # Extract features
    for i in range(1):
        features, _ = extract_features(model, train_dataloader_wag, num_features, device, verbose)
        logger.info('Get the deep feature')
        S = generate_wta_based_similarity(features, threshold=threshold)
        S = S.to(device)

        logger.info('END')
        

        # Training
        model.train()
        for epoch in range(max_iter):
            for i, (data, data_aug,_, index) in enumerate(train_dataloader):


                data = data.to(device)
                batch_size = data.shape[0]
                data_aug = data_aug.to(device)

                optimizer.zero_grad()

                v= model(data)
                v_aug= model(data_aug)
                h1 = F.normalize(v)
                h2 = F.normalize(v_aug)
                targets = S[index, :][:, index]
                codes = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)
                wta_contrast_loss = criterion1(codes, mask=targets)
                c = F.normalize(v,dim=0).T @ F.normalize(v_aug,dim=0)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
                off_diag = off_diagonal(c).pow(2).mean()
                bt_loss = on_diag + off_diag

                quan_loss = - torch.mean(torch.sum(F.normalize(h1,dim=-1)*F.normalize(torch.sign(h1),dim=-1), dim=1))

                loss = eta_2 * quan_loss + bt_loss * eta_1 + wta_contrast_loss
               

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
    #np.save("./code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())

    #np.save("./code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())

    #np.save("./code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())

    #np.save("./code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
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
            outputs= model(data)
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


    




def generate_wta_based_similarity(features, threshold):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        threshold: Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """

    number_permutation = 512


    features = features.numpy()
    # wta
    
    wta_model = WTA(
        K=4,
        number_of_permutations=1
    )

    
    new_features_c = np.zeros((features.shape[0], number_permutation))
    for i in range(number_permutation):
        theta = np.random.permutation(features.shape[1])
        new_features_one = wta_model.hash(features, theta)
        for j in range(features.shape[0]):
            new_features_c [j,i] = new_features_one[j].index(max(new_features_one[j]))
    
    #save the binary wta embedding
    #np.save('cifar_512.npy', new_features_c)

    number_training_samples = new_features_c.shape[0]

    new_features_c_tensor = torch.from_numpy(new_features_c)
    new_features_c_tensor1 = new_features_c_tensor.unsqueeze(1)
    new_features_c_tensor1 = new_features_c_tensor1.expand(number_training_samples, number_training_samples, new_features_c.shape[1])

    S = new_features_c_tensor1 - new_features_c_tensor
    S1 = torch.count_nonzero(S, dim=2)
    S2 = number_permutation - S1
    S3 = (S2 < threshold) * 0.0 + (S2 >= threshold) * 1.0

    return S3
