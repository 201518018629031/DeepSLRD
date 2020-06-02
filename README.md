# DeepSLRD
The implementation of our IJCNN 2019 paper "Deep Structure Learning for Rumor Detection on Twitter" [DeepSLRD](https://www.researchgate.net/publication/336169139_Deep_Structure_Learning_for_Rumor_Detection_on_Twitter).
# Requirements
python 3.6.6  
numpy==1.17.2  
networkx==2.2  
scipy==1.3.1
# How to use
## Dataset
unzip dataset.zip  
The dataset.zip includes nflod, resource, twitter15 and twitter16 folders. This dataset collected by Ma et al. (2018), and the raw datasets can be downloaded from [here](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0):  
Jing Ma, Wei Gao, Kam-Fai Wong. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018.  
The ind_twitter15.graph, ind_twitter15.features, ind_twitter15.poster, ind_twitter16.graph, ind_twitter16.features, and ind_twitter16.poster files are the propocessed data of user behavious graph on datasets twitter15 and twitter16, respectively.  
## Training & Testing
python Main_BU_RvNN_GCN.py #training and testing the BU-Hybrid model  
python Main_TD_RvNN_GCN.py #training and testing the TD-Hybrid model
# Citation
If you find the code is useful for your research, please cite this paper:  
<pre><code>@inproceedings{huang2019deep,
author = {Huang, Qi and Zhou, Chuan and Wu, Jia and Wang, Mingwen and Wang, Bin},
year = {2019},
month = {07},
pages = {1-8},
title = {Deep Structure Learning for Rumor Detection on Twitter},
doi = {10.1109/IJCNN.2019.8852468}
}</code></pre>
