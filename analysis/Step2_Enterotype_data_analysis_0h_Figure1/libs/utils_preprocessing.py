import numpy as np
import pandas as pd



def import_dataset(path, sort_feature=None, sort_value=None, sep=',', is_taxa_file=True, col_index='index'):
    # index feature 가 들어가야함

    df = pd.read_csv(path, sep=sep)

    if sort_feature != None:
        df = df[df[sort_feature] == sort_value]

    if is_taxa_file:
        taxa_total_columns = list(df.columns)
        taxa_features = [feat for feat in taxa_total_columns if 'd_' in feat]

        # metaindex
        meta_feauters = [item for item in taxa_total_columns if item not in taxa_features]
        #
        taxa_features = ['index'] + taxa_features

        df_taxa = df[taxa_features].set_index('index')
        df_meta = df[meta_feauters].set_index('index')


    else:
        df_taxa = df.set_index('index')
        df_meta = None
            

    return df_taxa, df_meta



def phylum_genus_name_change(df):
        
    # 피처 이름 변환 함수
    def simplify_feature_name(name):
        parts = name.split(';')
        new_name = []
        for part in parts:
            if part.startswith('p__') or part.startswith('g__'):
                new_name.append(part)
        return ';'.join(new_name)

    # 피처 이름 변환 적용
    new_columns = [simplify_feature_name(col) for col in df.columns]
    df_rev = df.copy()

    df_rev.columns = new_columns
    return df_rev


def remove_zero_feature(df, print_shape=True):
    # 모든 값이 0인 열의 이름을 식별
    cols_to_drop = [col for col in df.columns if (df[col] == 0).all()]

    # drop 메소드를 사용하여 열 제거  
    df_rev = df.drop(columns=cols_to_drop)

    if print_shape:
        print(f'original data shape: {df.shape}')
        print(f'returned data shape: {df_rev.shape}')

    return df_rev


def noise_removal(df, percent=0.01,):
    # 데이터프레임 복사
    matrix = df.copy()
    
    # 열 합계 계산
    col_sums = matrix.sum(axis=0)
    
    # 열 합계의 백분율 계산
    total_sum = col_sums.sum()
    col_percentages = (col_sums * 100) / total_sum
    
    # 특정 퍼센트 이상의 열 필터링
    big_ones = col_percentages > percent
    matrix_1 = matrix.loc[:, big_ones]
    return matrix_1



def relative_abundance(df):
    relative_abundance_df = df.div(df.sum(axis=1), axis=0)
    return relative_abundance_df


def feature_selection_sum(df, num:int):
    
    # featuer total sum
    feature_sums = df.sum(axis=0)

    #
    sorted_features = feature_sums.sort_values(ascending=False)[:num].index.tolist()

    #
    df_rev = df.copy()
    df_rev['Other'] = df.drop(columns=sorted_features).sum(axis=1)

    df_rev = df_rev[sorted_features + ['Other']]

    return df_rev


