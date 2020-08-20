from sklearn.decomposition import PCA
import numpy as np
data = np.array([[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]]) #data是一个(12,2)的array数组

pca=PCA(n_components=1)
#n_components表示所要保留的主成分个数n，也即保留下来的特征个数n  缺省时表示所有成分都被保留
#copy:bool值 表示是否改变原始数据 False表示改变原始数据 True表示不改变原始数据
#whiten:白化  使得每个特征具有相同的方差
newData=pca.fit_transform(data)