import numpy as np
import pandas as pd

def cluster_sorting(df, cluster_labels):
    '''
    cluster_labels = [0, 1, 2, 0, 0, 1] 이런 것

    df_sort = cluster_value 로 순서를 정렬 한것
    cluster_dict = cluster 를 sample 이름으로 변경
    '''
    df_rev = df.copy()
    df_rev['cluster_labels'] = cluster_labels
    df_sort_raw = df_rev.sort_values(by='cluster_labels')
    df_sort = df_sort_raw.drop(columns='cluster_labels')

    df_sort_raw = df_sort_raw.reset_index()
    cluster_dict = df_sort_raw.groupby('cluster_labels')['index'].apply(tuple).to_dict()
    return df_sort, df_sort_raw, cluster_dict, 



def nth_largest_feature(df, num):
    def num_large_feature(series, num=num-1):
        sorted_series = series.sort_values(ascending=False)
        return sorted_series.index[num]
    df_rev = df.copy()

    nth_dominant_features = df_rev.apply(num_large_feature, axis=1)
    return nth_dominant_features
