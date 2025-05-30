import ast
import csv
import json
import random
import torch
from utility.response_selection import keyword_based, vector_based
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
''' Read Data '''
def read_candidates(fname):
    candidates = []
    for line in open(fname, 'r'):
        candidates.append(line.strip())
    return candidates
 
 
def initialize_train_test_dataset(dataset):
    """ Create train and test dataset by random sampling.
        pct: percentage of training
    """
    pct = 0.80
    if dataset in ['reddit', 'gab']:
        dataset_fname = './data/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech-master/' + dataset + '.csv'
        xlist, ylist, zlist = read_EMNLP2019(dataset_fname)
        hate_num = 0
        for y in ylist:
            for i in y.strip('[]').split(', '):
                hate_num += 1
 
        X_text, Y_text = [], []
        line_num = 0
        for x, y, z in zip(xlist, ylist, zlist):
            x = x.strip().split('\n')
            for i in y.strip('[]').split(', '):
                X_text.append('. '.join(x[int(i) - 1].split('. ')[1:]).strip('\t'))  # Only the hate speech line.
                temp = []
                for j in split_response_func(z):
                    if j.lower() == 'n/a':
                        continue
                    temp.append(j)
                Y_text.append(temp)
                line_num += 1
    elif dataset == 'conan':
        all_text = [json.loads(line) for line in open('./data/CONAN/CONAN.json', 'r')]
        EN_text = [x for x in all_text[0]['conan'] if x['cn_id'][:2] == 'EN']
        X_text = [x['hateSpeech'].strip() for x in EN_text]
        Y_text = [[x['counterSpeech'].strip()] for x in EN_text]
        hate_num = len(X_text)
 
    elif dataset == 'sample':
        try:
            # Load the data using pandas
            df = pd.read_csv('/content/data/x_y_pairs.txt', sep='\t', header=None, names=['context', 'response'], on_bad_lines='skip', encoding='utf-8')
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
 
        # Drop rows with missing data
        df = df.dropna(subset=['context', 'response'])
 
        # Convert to lists
        X_text = df['context'].tolist()
        Y_text = df['response'].apply(lambda x: [x]).tolist()  # Keep responses as list of one item
 
        hate_num = len(X_text)
        pct = 0.8  # Or whatever percentage of training data you want
 
        import random
        all_indices = list(range(hate_num))
        train_indices = sorted(random.sample(all_indices, int(pct * hate_num)))
 
        train_x_text = [X_text[i] for i in train_indices]
        train_y_text = [Y_text[i] for i in train_indices]
 
        test_x_text = X_text  # Full dataset
        test_y_text = Y_text
 
        return train_x_text, train_y_text, test_x_text, test_y_text
 
 
 
    pct = 0.3  # percentage of data to use for training
 
    # Random subset for training
    import random
    all_indices = list(range(hate_num))
    train_indices = sorted(random.sample(all_indices, int(pct * hate_num)))
 
    # Build training subset
    train_x_text = [X_text[i] for i in train_indices]
    train_y_text = [Y_text[i] for i in train_indices]
 
    # Test set is the full dataset, in original order
    test_x_text = X_text
    test_y_text = Y_text
 
    return train_x_text, train_y_text, test_x_text, test_y_text
 
 
def read_EMNLP2019(dataset_fname):
    xlist = []
    ylist = []
    zlist = []
    with open(dataset_fname, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = row['text']
            y = row['hate_speech_idx']
            z = row['response']
            if y == 'n/a':
                continue
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
    return xlist, ylist, zlist
 
 
 
''' Data processing '''
def convert_to_contexts_responses(train_x_text, train_y_text):
    contexts_train = []
    responses_train = []
    for i in zip(train_x_text, train_y_text):
        contexts_train.append(i[0].strip())
        responses_train.append(i[1][random.randint(0, len(i[1])-1)].strip())
    return contexts_train, responses_train
 
 
 
''' Model '''
def to_method_object(method_name):
    method_name = method_name.upper()
    if method_name == 'TF_IDF':
        return keyword_based.TfIdfMethod()
    elif method_name == 'BM25':
        return keyword_based.BM25Method()
    elif method_name == 'USE_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder/2"))
    elif method_name == 'USE_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder/2"))
    elif method_name == 'USE_LARGE_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder-large/3"))
    elif method_name == 'USE_LARGE_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder-large/3"))
    elif method_name == 'ELMO_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/elmo/1"))
    elif method_name == 'ELMO_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/elmo/1"))
    elif method_name == 'BERT_SMALL_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.BERTEncoder("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"))
    elif method_name == 'BERT_SMALL_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.BERTEncoder("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"))
    elif method_name == 'BERT_LARGE_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.BERTEncoder("https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"))
    elif method_name == 'BERT_LARGE_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.BERTEncoder("https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"))
    elif method_name == 'USE_QA_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.USEDualEncoder("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"))
    elif method_name == 'USE_QA_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.USEDualEncoder("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"))
    elif method_name == 'CONVERT_SIM':
        return vector_based.VectorSimilarityMethod(encoder=vector_based.ConveRTEncoder("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2"))
    elif method_name == 'CONVERT_MAP':
        return vector_based.VectorMappingMethod(encoder=vector_based.ConveRTEncoder("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2"))
    raise ValueError("Unknown method {}".format(method_name))
 
 
 
''' Useful functions '''
def split_response_func(strresponse):
    result = ast.literal_eval(strresponse)
    result = [x for x in result]
    return result