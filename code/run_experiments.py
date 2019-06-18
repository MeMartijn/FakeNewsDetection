import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
import json
from keras.preprocessing import sequence

from data_loader import DataLoader
from classifiers import Classifiers

# Create class objects for interaction with data
data = DataLoader()
clfs = Classifiers()

# Load general data
liar = data.get_dfs()
liar = {
    fold: liar[fold][['label', 'statement']]
    for fold in liar.keys()
}

embedding_techniques = ['elmo', 'gpt', 'flair', 'bert', 'transformerxl']
clf_names = ['bilstm', 'cnn', 'svm', 'logres', 'gradientboosting']

# Create object to save results
results = {
    embedding: {
        label_count: {
            clf: {}
            for clf in clf_names
        }
        for label_count in [6, 3, 2]
    }
    for embedding in embedding_techniques
}

# Begin classifications
for embedding in tqdm(embedding_techniques):
    # Get the proper data
    df = eval('data.get_' + embedding + '()')
    general = deepcopy(liar)

    # Loop through each target variable count
    for label_count in [6, 3, 2]:
        if label_count == 3:
            # Recode labels from 6 to 3
            def recode(label):
                if label == 'false' or label == 'pants-fire' or label == 'barely-true':
                    return 'false'
                elif label == 'true' or label == 'mostly-true':
                    return 'true'
                elif label == 'half-true':
                    return 'half-true'

            for dataset in general.keys():
                general[dataset]['label'] = general[dataset]['label'].apply(lambda label: recode(label))
            
        elif label_count == 2:
            # Recode labels from 3 to 2
            def recode(label):
                if label == 'true' or label == 'half-true':
                    return 'true'
                elif label == 'false':
                    return 'false'

            for dataset in general.keys():
                general[dataset]['label'] = general[dataset]['label'].apply(lambda label: recode(label))
        
        # Test whether the amount of labels is correct
        assert (len(general['train']['label'].unique()) == label_count), 'The amount of labels is incorrect'

        # ------- START NEURAL CLASSIFICATIONS ------- #
        padding_variance = list(range(2, 37))
        results[embedding][label_count]['bilstm'] = {
            max_len: [] for max_len in padding_variance
        }
        results[embedding][label_count]['cnn'] = {
            max_len: [] for max_len in padding_variance
        }

        # Concatenate dataset
        concatenated_df = {
            fold: [np.concatenate(np.array(statement)) for statement in df[fold]['statement']]
            for fold in df.keys()
        }

        for max_len in padding_variance:
            # Apply padding
            padded_df = {
                fold: sequence.pad_sequences(concatenated_df[fold], maxlen = max_len, dtype = float)
                for fold in concatenated_df.keys()
            }

            # Get scores for neural classifiers
            for neural_net in ['bilstm', 'cnn']:
                for i in range(5):
                    test_score = eval("clfs.get_" + neural_net + "_score(padded_df['train'], padded_df['test'], padded_df['validation'], general['train']['label'], general['test']['label'], general['validation']['label'])")
                    results[embedding][label_count][neural_net][max_len].append(test_score)

            del padded_df
            
        # Clear up memory
        del concatenated_df

        # Update .json file
        with open('results.json', 'w') as fp:
            json.dump(results, fp)

        # ------- START LINEAR CLASSIFICATIONS ------- #
        linear_clfs = ['svm', 'logres', 'gradientboosting']
        pooling_techniques = ['max', 'min', 'average']
        regularizations = ['l1', 'l2']
        for linear_clf in linear_clfs:
                if linear_clf == 'gradientboosting':
                    results[embedding][label_count][linear_clf] = {
                        technique: [] for technique in pooling_techniques
                    }
                else: 
                    # There needs to be both a L1 version and a L2 version
                    results[embedding][label_count][linear_clf] = {
                        reg: {
                            technique: [] for technique in pooling_techniques
                        } for reg in regularizations
                    }
        
        for pooling_technique in pooling_techniques:
            # Apply pooling
            pooled_df = data.apply_pooling(pooling_technique, df)

            for linear_clf in linear_clfs:
                if linear_clf == 'gradientboosting':
                    for i in range(5):
                        test_score = eval("clfs.get_" + linear_clf + "_score(pooled_df['train'], pooled_df['test'], pooled_df['validation'], general['train']['label'], general['test']['label'], general['validation']['label'])")
                        results[embedding][label_count][linear_clf][pooling_technique].append(test_score)
                else:
                    for reg in regularizations:
                        for i in range(5):
                            test_score = eval("clfs.get_" + linear_clf + "_score(pooled_df['train'], pooled_df['test'], pooled_df['validation'], general['train']['label'], general['test']['label'], general['validation']['label'], penalty = '" + reg + "')")
                            results[embedding][label_count][linear_clf][reg][pooling_technique].append(test_score)
            
            del pooled_df
        
        # Update .json file
        with open('results.json', 'w') as fp:
            json.dump(results, fp)
