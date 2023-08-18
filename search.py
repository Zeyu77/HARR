import os
import sys
import time


def run_model(dataset, root, num_train, num_query, gpu, code_length,topk, num_class,threshold, eta, eta_1, eta_2, eta_3, batch_size, percentile, number_permutation, K, eta_4):
    cmd = "python run.py"
    set = cmd + " --dataset " + dataset + " --root " + root + " --num-query " + str(num_query) + " --num-train " + \
          str(num_train) + " --gpu " + gpu +" --train " +" --code-length " +str(code_length) + " --topk " +str(topk) + \
              " --threshold " +str(threshold) +  " --batch-size " +str(batch_size) + \
                  " --eta_1 " +str(eta_1) + " --eta_2 " +str(eta_2)+ " --eta_3 " +str(eta_3)+ " --percentile " +str(percentile)+ " --number_permutation " +str(number_permutation)+ " --K " +str(K)+ " --eta_4 " +str(eta_4)
    print(set)
    os.system(set)
    



if __name__ == '__main__':
    ind = int(sys.argv[1]) #which dataset to train
    gpu = sys.argv[2] # which gpu to use
    dataset_list = ['flickr25k', 'nus-wide-tc-10', 'cifar-10','nus-wide-tc-21', 'coco']
    dataset = dataset_list[ind]
    root_dict = {'flickr25k': '../../datasets/flickr25k', 'nus-wide-tc-10': '../../datasets/NUS-WIDE', 'cifar-10': '../../datasets','nus-wide-tc-21': '../../datasets/NUS-WIDE', 'coco': '../../datasets/coco'}
    root = root_dict[dataset]

    num_query_dict = {'flickr25k': 2000, 'nus-wide-tc-10': 5000, 'cifar-10':10000,'nus-wide-tc-21': 2100, 'coco':5000}
    num_query = num_query_dict[dataset]
    num_train_dict = {'flickr25k': 5000, 'nus-wide-tc-10': 5000, 'cifar-10':5000,'nus-wide-tc-21': 10500, 'coco':10000}
    num_train = num_train_dict[dataset]
    topk_dict = {'flickr25k': 5000, 'nus-wide-tc-10': 5000, 'cifar-10':-1,'nus-wide-tc-21': 5000, 'coco':5000}
    topk = topk_dict[dataset]
    code_length_list = [64]

    eta_list = [1]
    eta_1_list = [1]
    eta_2_list = [0.5]
    eta_3_list = []
    eta_3_list.append(ind)
    eta_4_list = [0, 1, 2, 3, 4, 5, 6, 7]
    batch_size = 64

    num_class_list=[70]

    threshold_list = [210]
    number_permutation_list = [512]
    K_list = [8]
    percentile_list = [0.3]



  
    print("start run all models")
    
    for code_length in code_length_list:
        for num_class in num_class_list:
            for threshold in threshold_list:
                for eta in eta_list:
                    for eta_1 in eta_1_list:
                        for eta_2 in eta_2_list:
                            for eta_3 in eta_3_list:
                                for number_permutation in number_permutation_list:
                                    for K in K_list:
                                        for percentile in percentile_list:
                                            for eta_4 in eta_4_list:
                                                time.sleep(2)
                                                run_model(dataset, root, num_train, num_query, gpu,code_length,topk, num_class, threshold,eta, eta_1, eta_2, eta_3, batch_size, percentile, number_permutation, K, eta_4)
