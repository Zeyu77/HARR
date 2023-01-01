# A pytorch implementation for model "HARR" 

## REQUIREMENTS
1. pytorch 1.10.0
2. loguru

## DATASETS
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## USAGE
```
HARR_Pytorch

optional arguments:
  -d DATASET, --dataset DATASET
                        Dataset name.
  -r ROOT, --root ROOT  Path of dataset
  -c CODE_LENGTH, --code-length CODE_LENGTH
                        Binary hash code length.(default: 64)
  -T MAX_ITER, --max-iter MAX_ITER
                        Number of iterations.(default: 800)
  -l LR, --lr LR        Learning rate.(default: 1e-3)
  -q NUM_QUERY, --num-query NUM_QUERY
                        Number of query data points.(default: 10000)
  -t NUM_TRAIN, --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 4)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.(default: 96)
  -k TOPK, --topk TOPK  Calculate map of top k.(default: -1)
  -v, --verbose         Print log.
  --train               Training mode.
  --evaluate            Evaluate mode.
  -g GPU, --gpu GPU     Using gpu.(default: 2)
  -e EVALUATE_INTERVAL, --evaluate-interval EVALUATE_INTERVAL
                        Interval of evaluation.(default: 50)
  --eta_1                 Hyper-parameter for weight balance. (default:1.0)
  --eta_2                 Hyper-parameter for weight balance. (default:0.5)

  --threshold             Hyper-parameter for similarity structure. (default:210)
  ```



## EXPERIMENTS
cifar10: 10000 query images, 5000 training images. Return MAP@ALL(50000) The same setting with DistillHash (19' CVPR )

How to train: python run.py --train 
