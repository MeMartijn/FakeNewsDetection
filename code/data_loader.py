import os
import pandas as pd

class DataLoader:
    '''A class which holds functionality to load and interact with the data from the research article'''
    def __init__(self):
        self.data_link = self.get_data_loc()
        self.df = self.get_dfs()
    
    def get_data_loc(self):
        '''Returns the file path for the data'''
        data_dir = 'Data'
        dir_name = 'liar_dataset'
        
        def ensemble_path(file):
            '''Short function to ensemble the paths of data files'''
            return data_dir + '/' + dir_name + '/' + file
    
        if dir_name in os.listdir('Data'):
            return {
                'train': ensemble_path('/train.tsv'),
                'test': ensemble_path('/test.tsv'),
                'validation': ensemble_path('/valid.tsv')
            }

        raise ValueError('Please unpack the ' + dir_name + '.zip file to continue')
    
    def get_dfs(self):
        '''Returns a dictionary with pandas dataframes for all data in the data_link dictionary'''
        return {
            dataset: pd.read_csv(self.data_link[dataset], sep='\t', header=None, index_col=0, names=['id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job', 'state', 'party', 'barely_true_count', 'false_count', 'half_true_count', 'mostly_true_count', 'pants_on_fire_count', 'context'])
            for dataset in self.data_link.keys()
        }
