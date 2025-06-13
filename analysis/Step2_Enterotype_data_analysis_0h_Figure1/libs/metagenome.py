import numpy as np
import pandas as pd
from skbio.stats.ordination import pcoa


# from utils_preprocessing import *
# from utils_figure import *
# from utils_distance import *


from libs.utils_preprocessing import *
from libs.utils_figure import *
from libs.utils_distance import *





class MetagenomeAalysis:
    def __init__(self, file_path=None, is_relative=True, sort_feature=None, sort_value=None, sep=',', is_taxa_file=True):
        self.file_path = file_path


        df, df_meta = import_dataset(self.file_path, sort_feature, sort_value, sep, is_taxa_file)

        df_sorted = remove_zero_feature(df)

        if is_relative:
            df_sorted = relative_abundance(df_sorted)

        self.df_raw = pd.read_csv(file_path, sep=sep)
        self.df = df 
        self.df_taxa_sorted = df_sorted
        self.df_meta = df_meta

    def change_phylum_genus_name(self, ):
        return phylum_genus_name_change(self.df_taxa_sorted)
    

    def feature_selection(self, num):
        return feature_selection_sum(self.df_taxa_sorted, num)
    
    def feature_remove_percent(self, percent=0.01):
        return noise_removal(self.df_taxa_sorted, percent)
    
    def cal_distance(self, distance):
        return calculate_distance(self.df_taxa_sorted, distance)
    


class MetagenomeFigure:

    def __init__(self, file_path=None, df=None, figsize=(5, 5), sep='\t'):

        if file_path != None:
            self.df = pd.read_csv(file_path, sep)

        elif type(df) == pd.core.frame.DataFrame:
            self.df = df

        else:
            ValueError('There is no file...')

        self.figsize = figsize
        self.features = df.columns

    def pam_clustering_eda(self, dist_matrix):
        nclusters_ch, nclusters_sil = calinski_harabasz_optimal_index(self.df, dist_matrix)
        return nclusters_ch, nclusters_sil

    def pam_clustering_k(self, dist_matrix, k):
        labels, clusters = pam_clustering(dist_matrix, k)
        return labels, clusters


    # PCoA 수행
    def get_pcoa_figure(self, dist_matrix, figsize=(4, 4), labels=None, is_sample_label=False, grid=False):

        pcoa_results = pcoa(dist_matrix)
        pcoa_df = pcoa_results.samples


        # 시각화
        plt.figure(figsize=figsize)

        # 시각화
        if type(labels) == np.ndarray:

            unique_clusters = np.unique(labels)

            for i, label in enumerate(unique_clusters):
                indices = np.where(np.array(labels) == label)
                plt.scatter(pcoa_df.iloc[indices]['PC1'], pcoa_df.iloc[indices]['PC2'], label=f'Cluster {label}')

        else:
            plt.scatter(pcoa_df['PC1'], pcoa_df['PC2'])

        
        if is_sample_label:
            for i in range(len(labels)):
                plt.text(pcoa_df.iloc[i]['PC1'], pcoa_df.iloc[i]['PC2'], labels[i], fontsize=9)

        plt.title('PCoA Result')
        plt.xlabel(f'PCoA1 ({pcoa_results.proportion_explained[0]*100:.2f}%)')
        plt.ylabel(f'PCoA2 ({pcoa_results.proportion_explained[1]*100:.2f}%)')
        plt.legend()
        plt.grid(grid)
        plt.show()

        return pcoa_results.samples



    def get_taxaplot(self, legend=True, order=True):
        taxa_plot(self.df, figsize=self.figsize, legend=legend, order=order)
        pass

    def get_group_taxaplot(self, ):
        pass


    
    def get_abundance(self, feature):
        pass

