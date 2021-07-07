# SDLM


This is author implementation of the paper:  
[Exploiting Syntactic Structure for Better Language Modeling: A Syntactic Distance Approach](https://arxiv.org/abs/2005.05864)  
Wenyu Du*, Zhouhan Lin*, Yikang Shen, Timothy J. O'Donnell, Yoshua Bengio, Yue Zhang (\* = equal contribution)  
ACL 2020 

The repository is originally forked from [Ordered Neuron (Shen et al. 2018)](https://github.com/yikangshen/Ordered-Neurons).
Please cite our work as follows if you use it in your work:
``` bibtex
@inproceedings{sdlm,
    title = "Exploiting Syntactic Structure for Better Language Modeling: A Syntactic Distance Approach",
    author = "Du, Wenyu  and Lin, Zhouhan  and  Shen, Yikang  and O{'}Donnell, Timothy J.  and  Bengio, Yoshua  and Zhang, Yue",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "6611--6628",
   }
```

## Usage

### Step 1: Prerequisite Software and Data Preparation

We simply replicate the environment in [Ordered Neuron (Shen et al. 2018)](https://github.com/yikangshen/Ordered-Neurons) with an update to PyTorch 1.2.

`data` consists of two parallel versions of PTB `penn` and `syntactic_penn`. `penn` is the vanilla Penn Treebank for language modelling, while `syntactic_penn` contains additional syntactic information.
You may still need to download the original PTB from [LDC](https://www.ldc.upenn.edu/) for testing.

### Step 2: Run the Experiments

Train language model with syntactic distance
``` bash
python main.py --l4d=0/1/2
```

Train language model without syntactic distance (Vanilla ON-LSTM LM)
``` bash
python main.py --un
```

### Step 3: Test

Decode using biased algorithm:
``` bash
python testf1.py  --b --l=0/1/2
```

Decode using unbiased algorithm:
``` bash
python testf1.py  --o --l=0/1/2
```