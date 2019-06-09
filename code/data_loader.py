import os
import pandas as pd
import numpy as np
import re
import nltk
from tqdm import tqdm

# Bag of Words imports
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer

# InferSent imports
import torch
from models import InferSent

# ELMo/BERT imports
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings, BertEmbeddings, TransformerXLEmbeddings, OpenAIGPTEmbeddings
import os
from nltk import tokenize

class FlairEncoder:
    '''An interface for interacting with Zalando's Flair library'''
    def __init__(self, embedding, data_dir, data):
        # Prepare tqdm loops
        tqdm.pandas()

        # Save variables to class
        self.embedding = embedding
        self.data_dir = data_dir
        self.dfs = data

    def get_embedded_dataset(self):
        '''Return the embedding representation of the dataset'''
        def get_embedding_dir(embedding):
            '''Turn the name of the embedding technique into a specific folder'''
            # Remove 'Embeddings' from the embedding name and lowercase it
            return embedding.__name__[:-10].lower()

        def create_embedding(statement, embedding):
            '''Create a single embedding from a piece of text'''
            # Split all sentences
            sentences = tokenize.sent_tokenize(statement)

            # Create an array for storing the embeddings
            vector = []

            # Loop over all sentences and apply embedding
            for sentence in sentences:
                # Create a Sentence object for each sentence in the statement
                sentence = Sentence(sentence, use_tokenizer = True)

                # Embed words in sentence
                embedding.embed(sentence)
                vector.append([token.embedding.numpy() for token in sentence])

            return vector
        
        def encode_datasets(embedding_dir, data_dir, dfs, embedding):
            '''Return all datasets with embeddings instead of texts'''
            # Check whether there already is a file containing the embeddings
            if embedding_dir in os.listdir(data_dir):
                # Return the previously made embeddings
                return {
                    dataset: pd.read_pickle(os.path.join(
                        data_dir, embedding_dir, dataset + '.pkl'
                    )) for dataset in dfs.keys()
                }
            else:
                print('Creating representations and saving them as files...')

                # Reformat dataframes to only contain the label and statement
                dfs = {
                    dataset:  dfs[dataset][['statement', 'label']]
                    for dataset in dfs.keys()
                }

                # Create a location to save the datasets as pickle files
                os.mkdir(os.path.join(data_dir, embedding_dir))

                # Activate embedding
                embedding = embedding()

                # Apply embedding
                for dataset in dfs:
                    # Apply transformation
                    dfs[dataset]['statement'] = dfs[dataset]['statement'].progress_map(
                        lambda text: create_embedding(text, embedding)
                    )

                    # Save the dataset as pickle file
                    file_path = os.path.join(data_dir, embedding_dir, dataset + '.pkl')
                    dfs[dataset].to_pickle(file_path)
                    print('Saved ' + dataset + '.pkl at ' + file_path)
                
                return dfs
        
        # Directory name for saving the datasets
        embedding_dir = get_embedding_dir(self.embedding)

        return encode_datasets(embedding_dir, self.data_dir, self.dfs, self.embedding)


class DataLoader:
    '''A class which holds functionality to load and interact with the data from the research article'''
    def __init__(self):
        # Prepare tqdm loops
        tqdm.pandas()

        # Location of the data directory
        self.data_dir = 'data'

        # Paths of the main dataset files
        self.data_link = self.get_data_loc()

        # Dataframes of the train, test and validation datasets
        self.df = self.get_dfs()

        # All the statements, used for training models
        self.all_statements = pd.concat([self.df[dataset] for dataset in self.df.keys()])['statement']

        # Set embedding functions
        self.set_embeddings()
    
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
        dfs = {
            dataset: pd.read_csv(self.data_link[dataset], sep='\t', header=None, index_col=0, names=['id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job', 'state', 'party', 'barely_true_count', 'false_count', 'half_true_count', 'mostly_true_count', 'pants_on_fire_count', 'context'])
            for dataset in self.data_link.keys()
        }

        # Data cleaning as described in the Data Explorations notebook
        for dataset in dfs.keys():
            dfs[dataset]['mistake_filter'] = dfs[dataset].statement.apply(lambda x: len(re.findall(r'\.json%%(mostly-true|true|half-true|false|barely-true|pants-fire)', re.sub(r'\t', '%%', x))))
            dfs[dataset] = dfs[dataset][dfs[dataset]['mistake_filter'] == 0]
            dfs[dataset].drop(columns='mistake_filter', inplace = True)
        
        return dfs

    def set_embeddings(self):
        '''Set all interfaces for embedding techniques using custom functions or Flair encoders'''
        def get_bow():
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
                        dataset: pd.read_pickle(os.path.join(
                            self.data_dir, bow_dir, dataset + '.pkl'))
                        for dataset in self.df.keys()
                    }
                else:
                    print('Creating Bag of Words representation and saving them as files...')
                    os.mkdir(self.data_dir + '/' + bow_dir)

                    # Create tokenizer
                    bow = CountVectorizer(
                        strip_accents='ascii',
                        analyzer='word',
                        tokenizer=tokenize,
                        stop_words='english',
                        lowercase=True,
                    )
                    bow.fit(self.all_statements)

                    # Get column names
                    features = bow.get_feature_names()
                    dfs = {
                        dataset: create_dataframe(self.df[dataset], bow, features)
                        for dataset in self.df.keys()
                    }

                    # Save the datasets as pickle files
                    for dataset in dfs.keys():
                        dfs[dataset].to_pickle(os.path.join(
                            self.data_dir, bow_dir, dataset + '.pkl'))
                    print('Saved the datasets at ' + bow_dir)

                    return dfs

            return init()
        
        def get_infersent():
            '''Returns InferSent representation of the dataset'''
            # Directory name for saving the datasets
            infersent_dir = 'infersent'
            fasttext_dir = 'fast-text'
            encoder_dir = 'encoder'

            def check_requirements():
                '''For training InferSent embeddings, a few data files must be present. This function checks these requirements. Returns either nothing or errors.'''
                # Make sure this is up to date
                nltk.download('punkt')

                if not fasttext_dir in os.listdir(self.data_dir):
                    print('Please refer to the InferSent installation instructions over here: https://github.com/facebookresearch/InferSent')
                    raise ValueError('Please download the fasttext vector file from https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip before continuing, and place the vector file (crawl-300d-2M.vec) in data/fast-text/.')

                if not encoder_dir in os.listdir():
                    print('Please refer to the InferSent installation instructions over here: https://github.com/facebookresearch/InferSent')
                    raise ValueError(
                        'Please run the following command and rerun this function: !curl -Lo encoder/infersent2.pickle https://dl.fbaipublicfiles.com/senteval/infersent/infersent2.pkl')

            def create_embedding(statement, encoder):
                '''Create an InferSent embedding from text'''
                sentences = tokenize.sent_tokenize(statement)
                return [encoder.encode(sentence, tokenize=True) for sentence in sentences]

            def create_dataframe(df, encoder):
                '''Create an InferSent dataframe from another dataframe'''
                new_df = df[['label', 'statement']]

                # Create a dataframe with the transformed statements
                new_df['statement'] = new_df['statement'].progress_map(
                    lambda statement: create_embedding(statement, encoder))

                return new_df

            def init():
                '''Initialize all logic from the main function'''
                # Check whether there is a file containing the InferSent data already present
                if infersent_dir in os.listdir(self.data_dir):
                    return {
                        dataset: pd.read_pickle(os.path.join(
                            self.data_dir, infersent_dir, dataset + '.pkl'))
                        for dataset in self.df.keys()
                    }
                else:
                    print('Creating InferSent representation and saving them as files...')

                    # Check whether model requirements are met
                    check_requirements()

                    # Load pre-trained model
                    model_version = 2
                    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
                    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
                    model = InferSent(params_model)
                    model.load_state_dict(torch.load(MODEL_PATH))

                    # Keep it on CPU or put it on GPU
                    use_cuda = False
                    model = model.cuda() if use_cuda else model

                    # Set word vectors
                    W2V_PATH = os.path.join(
                        self.data_dir, fasttext_dir, 'crawl-300d-2M.vec')
                    model.set_w2v_path(W2V_PATH)

                    # Build vocabulary
                    model.build_vocab(self.all_statements, tokenize=True)

                    dfs = {
                        dataset: create_dataframe(self.df[dataset], model)
                        for dataset in self.df.keys()
                    }

                    try:
                        # Create a storage directory
                        os.mkdir(self.data_dir + '/' + infersent_dir)

                        # Save the datasets as pickle files
                        for dataset in dfs.keys():
                            dfs[dataset].to_pickle(os.path.join(
                                self.data_dir, infersent_dir, dataset + '.pkl'))
                        print('Saved the datasets at ' + infersent_dir)

                        return dfs
                    except:
                        return dfs

            return init()

        # Attach all function references
        self.get_bow = get_bow
        self.get_infersent = get_infersent
        self.get_bert = FlairEncoder(BertEmbeddings, self.data_dir, self.df)
        self.get_elmo = FlairEncoder(ELMoEmbeddings, self.data_dir, self.df)
        self.get_transformerxl = FlairEncoder(TransformerXLEmbeddings, self.data_dir, self.df)
        self.get_gpt = FlairEncoder(OpenAIGPTEmbeddings, self.data_dir, self.df)
