import os
import pandas as pd
import numpy as np
import re
import nltk
from tqdm.auto import tqdm

# Bag of Words imports
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer

# InferSent imports
import torch
from models import InferSent

# Flair embedding imports
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings, BertEmbeddings, TransformerXLEmbeddings, OpenAIGPTEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, XLMEmbeddings, XLNetEmbeddings, OpenAIGPT2Embeddings
import os
from nltk import tokenize

# doc2vec
import gensim

class FlairEncoder:
    '''An interface for interacting with Zalando's Flair library'''
    def __init__(self, embedding, data_dir, data):
        # Prepare tqdm loops
        tqdm.pandas()

        # Save variables to class
        self.embedding = embedding
        self.data_dir = data_dir
        self.dfs = data

    def get_embedded_dataset(self, save = True):
        '''Return the embedding representation of the dataset'''
        def get_embedding_dir(embedding):
            '''Turn the name of the embedding technique into a specific folder'''
            if embedding[-1:] == ')':
                # The embedding technique is part of a function
                return re.search(r'"(.+)"', embedding).group(1)

            elif embedding[-10:] == 'Embeddings':
                # The embedding technique is part of the Flair library
                return embedding.split('Embeddings')[0].lower()
            
            else:
                raise ValueError('The requested embedding type is not supported by the data loader.')

        def create_embedding(statement, embedding):
            '''Create a single embedding from a piece of text'''
            try:
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
            except:
                print(statement)
        
        def encode_datasets(embedding_dir, data_dir, dfs, embedding):
            '''Return all datasets with embeddings instead of texts'''
            # Check whether there already is a file containing the embeddings
            if embedding_dir in os.listdir(data_dir):
                if embedding_dir == 'flair':
                    # Flair's training set is divided into two pickle files
                    return {
                        dataset: (pd.read_pickle(os.path.join(data_dir, embedding_dir, dataset + '.pkl')) 
                            if dataset != 'train' else 
                            pd.concat([
                                pd.read_pickle(os.path.join(data_dir, embedding_dir, dataset + '_subset1.pkl')),
                                pd.read_pickle(os.path.join(data_dir, embedding_dir, dataset + '_subset2.pkl'))
                                ]
                            ))
                        for dataset in dfs.keys()
                    }
                else:
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

                # Activate embedding
                if embedding_dir == 'flair':
                    # Flair's recommended usage is different from other embedding techniques
                    embedding = StackedEmbeddings([
                        WordEmbeddings('glove'),
                        FlairEmbeddings('news-forward'),
                        FlairEmbeddings('news-backward'),
                    ])
                elif embedding[-1:] == ')':
                    # This embedding has parameters
                    embedding = eval(embedding)
                else:
                    embedding = eval(embedding + '()')

                # Apply embedding
                for dataset in dfs:
                    # Apply transformation
                    dfs[dataset]['statement'] = dfs[dataset]['statement'].progress_map(
                        lambda text: create_embedding(text, embedding)
                    )

                    if embedding_dir == 'flair' and dataset == 'train' and save:
                        # Flair produces files too large: we need to split them before being able to save as files
                        total_length = len(dfs[dataset])
                        split = round(total_length / 2)
                        flair_subset1 = dfs[dataset].iloc[0:split]
                        flair_subset2 = dfs[dataset].iloc[split:]

                        # Save both as files
                        flair_subset1.to_pickle(os.path.join(data_dir, embedding_dir, dataset + '_subset1.pkl'))
                        flair_subset2.to_pickle(os.path.join(data_dir, embedding_dir, dataset + '_subset2.pkl'))

                        print('Because of the file size, the training set has been split and saved in two seperate files.')
                    elif save:
                        if embedding_dir not in os.listdir(data_dir):
                            # Create a location to save the datasets as pickle files
                            os.mkdir(os.path.join(data_dir, embedding_dir))

                        if dataset == 'train':
                            # Some tokenizers fail on these two statements
                            dfs[dataset].loc['3561.json', 'statement'] = create_embedding('Says JoAnne Kloppenburgs side had a 3-to-1 money advantage in the Wisconsin Supreme Court campaign.', embedding)
                            dfs[dataset].loc['4675.json', 'statement'] = create_embedding('Since Mayor Kennedy OBrien took office Sayreville has issued 22081 building permits! Now OBrien is holding secret meetings with big developers.', embedding)

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
        def get_bow(save = False):
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
                    if save:
                        for dataset in dfs.keys():
                            dfs[dataset].to_pickle(os.path.join(
                                self.data_dir, bow_dir, dataset + '.pkl'))
                        print('Saved the datasets at ' + bow_dir)

                    return dfs

            return init()
        
        def get_infersent(save = False):
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
                        if save:
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

        def get_flair_embedding(embedding, data_dir, dfs):
            encoder = FlairEncoder(embedding, self.data_dir, self.df)
            return encoder.get_embedded_dataset

        def get_doc2vec():
            '''Returns doc2vec representation of the dataset'''
            # Directory name for saving the datasets
            dataset_dir = 'doc2vec'

            def create_embeddings(df):
                '''Create doc2vec embeddings from dataframe'''
                # Create a training corpus
                train_corpus = [gensim.models.doc2vec.TaggedDocument(row.statement, [index]) for index, row in df['train'].iterrows()]

                # Set model parameters
                model = gensim.models.doc2vec.Doc2Vec(vector_size = 4600, min_count = 2, epochs = 40)

                # Build the vocabulary
                model.build_vocab(train_corpus)

                # Train the model
                model.train(train_corpus, total_examples = model.corpus_count, epochs = model.epochs)

                # Apply model to all statements
                embedded_df = df.copy()
                for dataset in df:
                    embedded_df[dataset]['statement'] = df[dataset]['statement'].apply(lambda statement: model.infer_vector(statement))
                
                return embedded_df

            def create_dataframe(df):
                '''Create an doc2vec dataframe from another dataframe'''
                doc2vec = {}

                for dataset in self.df.keys():
                    # Reduce columns
                    doc2vec[dataset] = self.df[dataset][['label', 'statement']]

                    # Preprocess statements
                    doc2vec[dataset]['statement'] = doc2vec[dataset]['statement'].map(lambda statement: gensim.utils.simple_preprocess(statement))
                
                return doc2vec

            def save_dataframes(dfs, target_dir):
                '''Save dataframes at the specified location'''
                # Create a storage directory
                os.mkdir(os.path.join(self.data_dir, target_dir))
                
                # Save the datasets as pickle files
                for dataset in dfs.keys():
                    dfs[dataset].to_pickle(os.path.join(
                        self.data_dir, target_dir, dataset + '.pkl'))
                print('Saved the datasets at ' + target_dir)

            def init():
                '''Initialize all logic from the main function'''
                # Check whether there is a file containing the doc2vec data already present
                if dataset_dir in os.listdir(self.data_dir):
                    return {
                        dataset: pd.read_pickle(os.path.join(
                            self.data_dir, dataset_dir, dataset + '.pkl'))
                        for dataset in self.df.keys()
                    }
                else:
                    print('Creating doc2vec representation and saving them as files...')

                    # Apply transformations to dataframe
                    doc2vec = create_dataframe(self.df)
                    doc2vec = create_embeddings(doc2vec)

                    # Save the dataframes as pickle files for later use
                    save_dataframes(doc2vec, dataset_dir)

                    return doc2vec
                
            return init()

        # Attach all function references
        self.get_bow = get_bow
        self.get_infersent = get_infersent
        self.get_bert = get_flair_embedding('BertEmbeddings', self.data_dir, self.df)
        self.get_elmo = get_flair_embedding('ELMoEmbeddings', self.data_dir, self.df)
        self.get_transformerxl = get_flair_embedding('TransformerXLEmbeddings', self.data_dir, self.df)
        self.get_gpt = get_flair_embedding('OpenAIGPTEmbeddings', self.data_dir, self.df)
        self.get_flair = get_flair_embedding('FlairEmbeddings', self.data_dir, self.df)
        self.get_fasttext = get_flair_embedding('WordEmbeddings("en-crawl")', self.data_dir, self.df)
        self.get_doc2vec = get_doc2vec
        self.get_gpt2 = get_flair_embedding('OpenAIGPT2Embeddings', self.data_dir, self.df)
        self.get_xlm = get_flair_embedding('XLMEmbeddings', self.data_dir, self.df)
        self.get_xlnet = get_flair_embedding('XLNetEmbeddings', self.data_dir, self.df)
    
    @staticmethod
    def apply_pooling(technique, df):
        '''Functionality to apply a pooling technique to a dataframe'''
        def pooling(vector):
            if technique == 'max':
                # Max pooling
                if len(vector) > 1:
                    return [row.max() for row in np.transpose([[token_row.max() for token_row in np.transpose(np.array(sentence))] for sentence in vector])]
                else:
                    return [token_row.max() for token_row in np.transpose(vector[0])]
            elif technique == 'min':
                # Min pooling
                if len(vector) > 1:
                    return [row.min() for row in np.transpose([[token_row.min() for token_row in np.transpose(np.array(sentence))] for sentence in vector])]
                else:
                    return [token_row.min() for token_row in np.transpose(vector[0])]
            elif technique == 'average':
                # Average pooling
                if len(vector) > 1:
                    return [np.average(row) for row in np.transpose([[np.average(token_row) for token_row in np.transpose(np.array(sentence))] for sentence in vector])]
                else:
                    return [np.average(token_row) for token_row in np.transpose(vector[0])]
            else:
                raise ValueError('This pooling technique has not been implemented. Please only use \'min\', \'max\' or \'average\' as keywords.')

        def init():
            '''Execute all logic'''
            print('Applying ' + technique + ' pooling to the dataset...')
            return {
                dataset: list(
                        df[dataset].statement.progress_apply(
                            lambda statement: pooling(statement)
                        ).values
                    )
                for dataset in df.keys()
            }

        return init()
