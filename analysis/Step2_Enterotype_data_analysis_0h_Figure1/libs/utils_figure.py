import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from pyclustering.cluster.kmedoids import kmedoids

from skbio.stats.ordination import pcoa

from sklearn.metrics import silhouette_score, calinski_harabasz_score




# 색 조합 1 (무지개 - 하늘색/청록색/연두색/노란색/주황색 .... 총 30색)
cmap1 = colors.ListedColormap(['#b6e1f3','#1699a8','#aad356','#f9c908','#f25844','#c44065','#d9c097','#fff9d6','#a6e0b9','#00ada7',
                               '#183c59','#735a99','#ff4062','#ff9c98','#facdaa','#c8c9a8','#80b09b','#c5c7c7','#007792','#011519',
                               '#8c296b','#897800','#da8e00','#e05500','#920000','#4c5760','#93a8ac','#bab7d3','#a59e8c','#66635b'])

# 색 조합 2 (파스텔 - 연분홍색/연주황색/겨자색/옥색/하늘색 .... 총 30색)
cmap2 = colors.ListedColormap(['#fbd0c0','#f3b988','#e1c784','#87dfbb','#61eaec','#4ac2e6','#b3a3c5','#5ebbb6','#88e0bb','#a1ca8e',
                               '#e6efab','#cce8d6','#fff2cc','#fbc6d6','#fbc6d6','#ee9bb9','#f8cbad','#ffe699','#c5e0b4','#7bc19c',
                               '#6ba599','#3abce4','#1b8c95','#975f8c','#a76d5d','#d0a292','#d872a0','#cab4cc','#d9d9d9','#7f7f7f'])

# 색 조합 3 (사계절 - 짙은청색/민트색/옅은녹색/아이보리색/살구색/주황색 .... 총 30색)
cmap3 = colors.ListedColormap(['#12263a','#06bcc1','#c5d8d1','#f4edea','#f4d1ae','#f28f3b','#c8553d','#985277','#784863','#847e89',
                               '#afbfc0','#cce3de','#a7c957','#6a994e','#386641','#677423','#bbc191','#e8e0d0','#b47b54','#845b3c',
                               '#bfa4a4','#e6c0c3','#ffa29b','#fb6376','#b44b2e','#d87b0a','#eeba0b','#f4e409','#ecf49e','#7f7f7f'])




### 아직 좀 더 다듬어야 함...

def initialize_medoids(n_samples, n_clusters):
    '''
    clustering 할 때 초창기 어디서 부터 시작할 것인지, 위치 잡는 것...
    
    '''
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    return indices.tolist()


def pam_clustering(distance_matrix, k):
    '''
    distance_matrix = 거리 계산된 matrix
    k = 몇개로 clustering 할지

    return labels = e.g. CH value 평가 할때
    array([0, 0, 1, 0, 0, 1, 2, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 0, 2, 0, 0, 2, 0, 2, 2, 2, 2])

    return clusters = e.g. 묶인 clustser = visualiation...
    [[6, 12, 24, 27, 29, 30, 31, 32],
    [0, 1, 3, 4, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 28],
    [2, 5, 7, 8, 15, 21]]
    '''
    initial_medoids = initialize_medoids(distance_matrix.shape[0], k)
    kmedoids_instance = kmedoids(distance_matrix.values, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    labels = np.zeros(distance_matrix.shape[0], dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
    return labels, clusters

def calinski_harabasz_optimal_index(df, dist_matrix):
    # 클러스터 수 평가s
    nclusters = [None]  # nclusters를 초기화
    nclusters_sil = [None]
    for k in range(2, len(df.index)):  # 2부터 20까지 클러스터 개수에 대해 평가
        labels, clusters = pam_clustering(dist_matrix, k)
        ch_index = calinski_harabasz_score(df, labels)
        nclusters.append(ch_index)

        silhouette_avg = silhouette_score(df, labels)
        nclusters_sil.append(silhouette_avg)

    # 결과 출력
    print("Calinski-Harabasz Index for different cluster numbers:")
    for k in range(2, len(df.index)):
        print(f"k={k}: CH Index={nclusters[k-1]}")

    return nclusters, nclusters_sil


def silhouette_score_index(df, clusters):
    silhouette_avg = silhouette_score(df, clusters)
    return silhouette_avg


def taxa_plot(df, figsize=(6, 6), width=0.8, linewidth=0, colormap=cmap3, legend=True, order=True):

    # 시각화
    if order != True:
        df = df.reindex(order)

    df.plot(kind='bar', stacked=True, figsize=figsize, width=width, linewidth=linewidth, colormap=colormap)
    plt.xlabel('Sample')
    plt.ylabel('Relative Abundance')
    if legend == True:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def plot_vertical(values, figsize):
    '''
    첫 값이 은 제외하고 ploting...
    '''

    fig, ax = plt.subplots(figsize=figsize)
    max_value = max(values[1:])
    max_index = values.index(max_value)


    for i, value in enumerate(values):
        ax.vlines(x=i, ymin=0, ymax=value, color='b', linestyle='--', linewidth=2)

    # 각 포인트에 점 추가
    ax.scatter(max_index, max_value, color='r', s=30, zorder=5)

    # 축 라벨 설정
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Vertical Lines Plot')

    # 그래프 표시
    plt.show()