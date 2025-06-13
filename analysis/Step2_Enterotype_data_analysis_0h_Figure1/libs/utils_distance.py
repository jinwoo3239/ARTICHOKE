import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, jensenshannon




def calculate_distance(df, distance, pesudocount=0.000001):

    # qiime2 에서 만든 것으로 가는 것이 가장 좋음...

    df[df == 0] = 0.000001


    if distance  == 'bray_curtis':
        # 브레이-커티스 불일치 지수 계산
        bray_curtis_distances = pdist(df.values, metric='braycurtis')
        dist_matrix = squareform(bray_curtis_distances)
        bray_curtis_matrix = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
        return bray_curtis_matrix

    elif distance == 'jaccard':
        # 자카드 불일치 지수 계산
        jaccard_distances = pdist(df.values, metric='jaccard')
        dist_matrix = squareform(jaccard_distances)

        jaccard_matrix = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

        return jaccard_matrix
    
    

    elif distance == 'jensen':
        jessen_matrix = dist_JSD(df, pseudocount=pesudocount)
        return jessen_matrix


        # 원래 아래처럼 코드를 짯으나... 이게 결과 값이 tutorial 에서 나오는 것 과 다름... 왜 다른지를 모르겠음...
        # # 데이터 간의 Jensen-Shannon Divergence 거리 행렬 계산
        # dist_matrix = np.zeros((df.shape[0], df.shape[0]))

        # for i in range(df.shape[0]):
        #     for j in range(df.shape[0]):
        #         if i != j:
        #             dist_matrix[i, j] = jensenshannon(df.iloc[i], df.iloc[j])

        # jensen_matrix = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
        # return jensen_matrix
    
    else:
        raise ValueError("bray_curtis, jaccard, jensen")



def dist_JSD(df, pseudocount=1e-6):
    '''
    enterotype tutorial 에 나온 것과 동일한 결과가 나올 수 있도록 만듬
    이상하게.. 왜 다르지?? 어쨋든 tutorial 에서 나오는 것 처럼 만들었으니 이를 가지고 쓰면 될듯...
    '''
    def KLD(x, y):
        return np.sum(x * np.log(x / y))

    def JSD(x, y):
        m = (x + y) / 2
        return np.sqrt(0.5 * KLD(x, m) + 0.5 * KLD(y, m))

    df[df == 0] = pseudocount

    matrixColSize = df.shape[0]
    resultsMatrix = np.zeros((matrixColSize, matrixColSize))

    for i in range(matrixColSize):
        for j in range(matrixColSize):
            resultsMatrix[i, j] = JSD(df.iloc[i], df.iloc[j])

    return pd.DataFrame(resultsMatrix, 
                        columns=df.index, 
                        index=df.index)


