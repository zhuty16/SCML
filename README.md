# Social Collaborative Mutual Learning
This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Guannan Liu, and Guoqing Chen. "Social Collaborative Mutual Learning for Item Recommendation." ACM Transactions on Knowledge Discovery from Data (TKDD) 14.4 (2020): 1-19.

## Introduction
Social Collaborative Mutual Learning (SCML) is a social recommendation framework that combines the item-based CF model with the social CF model by two mutual regularization strategies.

![](https://github.com/zhuty16/SCML/blob/master/framework.jpg)

## Citation
```
@article{zhu2020social,
  title = {Social Collaborative Mutual Learning for Item Recommendation},
  author = {Tianyu Zhu and Guannan Liu and Guoqing Chen},
  journal = {ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume = {14},
  number = {4},
  pages = {1--19},
  year = {2020},
  publisher = {ACM New York, NY, USA}
}
```

## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 1.5.0
* numpy == 1.14.2
* scipy == 1.1.0

## Example to Run the Codes
* Ciao dataset
```
python main.py --dataset=Ciao
```

