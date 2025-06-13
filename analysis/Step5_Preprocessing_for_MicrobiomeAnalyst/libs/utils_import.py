import pandas as pd






# from taxa level

def import_dataset(path, sort_feature=None, sort_value=None, sep=',', is_taxa_file=True, is_metabbiomeanalyst=False, save_file_name='data'):
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

        df_asv = df[taxa_features].set_index('index')
        df_meta = df[meta_feauters].set_index('index')


    else:
        df_asv = df.set_index('index')
        df_meta = None

    print(f'df_asv: {df_asv.shape}')
    print(f'df_meta: {df_meta.shape}')

    if is_metabbiomeanalyst:
        df_asv = df_asv.T
        df_asv.index.name = '#NAME'

        df_meta.index.name = '#NAME'


        print('save_file...')
        df_asv.to_csv(f'./{save_file_name}_asv.csv')
        df_meta.to_csv(f'./{save_file_name}_meta.csv')

        return df_asv, df_meta

    return df_asv, df_meta




# from raw_data

def preprocessing_taxa(path, sep='\t', feature_id='Feature ID', taxa_colum='Taxon', is_save=True, file_name='processed_taxonomy'):
    df = pd.read_csv(path, sep=sep)
    
    df_taxa_id = df[feature_id]

    #Taxon
    df_split = df.Taxon.str.split(';', expand=True)
    df_split = df_split.apply(lambda x: x.str.split('__').str[1])    
    df_split.columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    df_merge = pd.concat([df_taxa_id, df_split], axis=1)
    df_merge = df_merge.set_index(feature_id)

    df_merge.index.name = '#TAXONOMY'

    if is_save:
        print('file_save...')
        df_merge.to_csv(f'./{file_name}.csv')

    return df_merge


def sort_asv_table(path_asv, asv_sep, path_meta, meta_sep, asv_index, meta_index, sort_feature, sort_value, is_save=True, save_file_name='sort_file'):

    df_asv = pd.read_csv(path_asv, sep=asv_sep)
    df_asv = df_asv.set_index(asv_index).T


    df_meta = pd.read_csv(path_meta, sep=meta_sep)
    df_meta = df_meta.set_index(meta_index)
    df_meta.index.name = asv_index

    # column_name
    df_asv_columns = df_asv.columns
    df_meta_columns = df_meta.columns

    # merge
    df_merge = df_asv.join(df_meta, how='inner')

    df_sort = df_merge[df_merge[sort_feature] == sort_value]

    df_asv_rev = df_sort[df_asv_columns]
    df_meta_rev = df_sort[df_meta_columns]

    df_asv_rev.index.name = '#NAME'
    df_meta_rev.index.name = '#NAME'

    df_asv_rev = df_asv_rev.T
    df_asv_rev.index.name = '#NAME'

    if is_save:
        print('file_save...')
        df_asv_rev.to_csv(f'./{save_file_name}_asv.csv')
        df_meta_rev.to_csv(f'./{save_file_name}_meta.csv')

    return df_asv_rev, df_meta_rev