import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer

class DataLoader:
    '''A class which holds functionality to load and interact with the data from the research article'''
    def __init__(self):
        # Location of the data directory
        self.data_dir = 'data'

        # Paths of the main dataset files
        self.data_link = self.get_data_loc()

        # Dataframes of the train, test and validation datasets
        self.df = self.get_dfs()
    
    def get_data_loc(self):
        '''Returns the file path for the data'''
        dir_name = 'liar_dataset'
        
        def ensemble_path(file):
            '''Short function to ensemble the paths of data files'''
            return self.data_dir + '/' + dir_name + '/' + file
    
        if dir_name in os.listdir(self.data_dir):
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

    def get_bow(self):
        '''Returns bag of words representation of the dataset'''
        # Directory name for saving the datasets
        bow_dir = 'bag-of-words'

        def tokenize(statement):
            '''Tokenize function for CountVectorizer'''
            # Tokenize, lowercase
            statement = word_tokenize(statement.lower())

            # Remove punctuation
            table = str.maketrans('', '', string.punctuation)
            statement = [word.translate(table) for word in statement]

            # Remove empty strings
            statement = [word for word in statement if len(word) > 0]

            # Apply stemming
            porter = PorterStemmer()
            statement = [porter.stem(word) for word in statement]

            return statement

        def create_dataframe(df, vectorizer, features):
            '''Create a Bag of Words dataframe from another dataframe'''
            # Create a dataframe from the transformed statements
            new_df = pd.DataFrame(vectorizer.transform(
                df.statement).toarray(), columns=features)

            # Add referencing columns
            new_df['label'] = list(df.label)
            new_df['id'] = list(df.index)
            new_df.set_index('id', inplace=True)

            return new_df

        def init():
            '''Initialize all logic from the main function'''
            # Check whether there is a file containing the BoW data already present
            if bow_dir in os.listdir(self.data_dir):
                return {
                    dataset: pd.read_pickle(os.path.join(self.data_dir, bow_dir, dataset + '.pkl'))
                    for dataset in self.df.keys()
                }
            else:
                print('Creating Bag of Words representation and saving them as files...')
                os.mkdir(self.data_dir + '/' + bow_dir)

                # Combine all statements to extract all possible features
                all_statements = pd.concat([self.df[dataset] for dataset in self.df.keys()])['statement']

                # Create tokenizer
                bow = CountVectorizer(
                    strip_accents = 'ascii',
                    analyzer = 'word',
                    tokenizer = tokenize,
                    stop_words = 'english',
                    lowercase = True,
                )
                bow.fit(all_statements)

                # Get column names
                features = bow.get_feature_names()
                dfs = {
                    dataset: create_dataframe(self.df[dataset], bow, features)
                    for dataset in self.df.keys()
                }

                # Save the datasets as pickle files
                for dataset in dfs.keys():
                    dfs[dataset].to_pickle(os.path.join(self.data_dir, bow_dir, dataset + '.pkl'))
                print('Saved the datasets at ' + bow_dir)

                return dfs

        return init()
