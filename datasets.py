import abc

import pandas as pd
import numpy as np
import glob


class Dataset(abc.ABC):
    df: pd.DataFrame
    df_train: pd.DataFrame
    df_eval: pd.DataFrame
    df_test: pd.DataFrame

    def run(self):
        pass

    def statistics(self):
        pass

    @abc.abstractmethod
    def name(self):
        pass

    def size(self):
        # use after split
        return int(len(self.df))

    @abc.abstractmethod
    def label(self):
        pass

    def split(self):
        train_size = 0.8
        eval_size = 0.1

        train_index = int(len(self.df)*train_size)
        eval_index = int(len(self.df)*eval_size)

        self.df_train = self.df[0:train_index]
        self.df_eval = self.df[train_index:train_index+eval_index]
        self.df_test = self.df[train_index+eval_index:]

    def limit(self):
        self.df_train = self.df_train[:1024]
        self.df_eval = self.df_eval[:32]
        self.df_test = self.df_test[:32]


class DanishTweetsDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('/home/aleksander/.danlp/twitter.sentiment.csv')
        self.df = self.df[['polarity', 'text']]
        self.df = self.df.rename(columns={'polarity': 'label'})

    def name(self):
        return "danish-tweets"

    def label(self):
        return ['positiv', 'neutral', 'negativ']


class AmazonDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('data/amazon_review_polarity_csv/train.csv', header=None)
        self.df = self.df.drop(self.df.columns[[1]], axis=1)
        self.df.columns = ['label', 'text']
        self.df = self.df.sample(int(self.df.shape[0] * 0.01), random_state=42)

        self.df_test = pd.read_csv('data/amazon_review_polarity_csv/test.csv', header=None)
        self.df_test = self.df_test.drop(self.df_test.columns[[1]], axis=1)
        self.df_test.columns = ['label', 'text']
        self.df_test = self.df_test.sample(int(self.df_test.shape[0] * 0.01), random_state=42)

    def size(self):
        return int(len(self.df)+len(self.df_test))

    def name(self):
        return "amazon-polarity"

    def label(self):
        return [1, 2]

    def split(self):
        train_size = int(len(self.df))
        eval_size = int(len(self.df_test))

        self.df_train = self.df[0:(train_size - eval_size)]
        self.df_eval = self.df[(train_size-eval_size):train_size]


class YelpDataset(Dataset):
    def __init__(self):
        self.df = pd.DataFrame()
        chunks = pd.read_json('data/yelp/yelp_academic_dataset_review.json',
                              lines=True,
                              chunksize=1000000)
        for chunk in chunks:
            self.df = pd.concat([self.df, chunk])

        self.df = self.df[['stars', 'text']]
        self.df = self.df.rename(columns={'stars': 'label'})
        self.df = self.df.sample(int(self.df.shape[0] * 0.01), random_state=42)

    def name(self):
        return "yelp"

    def label(self):
        return np.arange(1.0, 5.5, 0.5)


class IMDbDataset(Dataset):
    def __init__(self):
        self.df = pd.DataFrame(columns=['label', 'text'])

        switch_to = False
        complete = False

        while complete is False:

            if switch_to is False:
                filepath = "data/imdb/train/pos/*"
            else:
                filepath = "data/imdb/train/neg/*"

            for file in glob.glob(filepath):
                my_file = open(file, "r")
                lines = my_file.read()

                index = file.rfind('/')
                file = file[index:len(file)]
                index = file.rfind('_')
                file = file[(index + 1):len(file)]
                index = file.rfind('.')
                file = file[0:index]

                self.df = self.df.append({'label': int(file), 'text': lines}, ignore_index=True)

            if switch_to is True:
                complete = True
            switch_to = True

        for i in range(5):
            self.df = self.df.sample(frac=1).reset_index()
            self.df = self.df.drop('index', axis=1)

    def name(self):
        return "imdb"

    def label(self):
        return range(1, 11)


class PolEmoDataset(Dataset):
    def __init__(self):
        pass

    def name(self):
        return "polemo"

    def size(self):
        # use after split
        return int(len(self.df_train)+len(self.df_eval)+len(self.df_test))

    def label(self):
        return ['minus_m', 'plus_m', 'amb', 'zero']

    def read(self, name, attribute):
        filepath = 'data/polemo/{}.text.{}.txt'.format(name, attribute)
        if attribute == "train":
            self.df_train = pd.DataFrame(open(filepath).readlines())
            self.process(self.df_train)
        elif attribute == "dev":
            self.df_eval = pd.DataFrame(open(filepath).readlines())
            self.process(self.df_eval)
        elif attribute == "test":
            self.df_test = pd.DataFrame(open(filepath).readlines())
            self.process(self.df_test)

    @staticmethod
    def process(dataframe: pd.DataFrame):
        dataframe.columns = ['text']
        dataframe['label'] = ""
        for i in range(len(dataframe)):
            dataframe['label'][i] = dataframe['text'][i].split('__label__meta_')[1].strip()
            dataframe['text'][i] = dataframe['text'][i].split(' __label__meta_')[0]

    def split(self):
        pass


class ClarinEmoDataset(Dataset):
    def __init__(self):
        self.df_train = pd.read_csv('data/clarinemo/train/PolEmoAggregatedTrain.csv')
        self.process(self.df_train)
        self.df_eval = pd.read_csv('data/clarinemo/val/PolEmoAggregatedVal.csv')
        self.process(self.df_eval)
        self.df_test = pd.read_csv('data/clarinemo/test/PolEmoAggregatedTest.csv')
        self.process(self.df_test)

    def name(self):
        return "clarinemo"

    def label(self):
        return ['pozytywny', 'negatywny', 'neutralny', 'brak']

    def split(self):
        pass

    def process(self, dataframe: pd.DataFrame):
        dataframe['label'] = ""
        for i in range(len(dataframe)):
            for lab in self.label():
                if lab is not 'brak' and dataframe[lab][i]:
                    dataframe['label'][i] = lab
            if dataframe['label'][i] not in self.label():
                dataframe['label'][i] = 'brak'


class ClarinEmoEmoDataset(ClarinEmoDataset):
    def name(self):
        return "clarinemo-emo"

    def label(self):
        return ['radość', 'zaufanie', 'przeczuwanie', 'zdziwienie', 'strach', 'smutek', 'wstręt', 'gniew', 'brak']


class CawiTwoDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('data/cawi2/cawi2_selected_annotations.csv')
        self.df_texts = pd.read_csv('data/cawi2/cawi2_selected_texts.csv')
        self.process(self.df)
        self.process_texts(self.df, self.df_texts)

    def name(self):
        return "cawi2"

    def label(self):
        return ['pozytywny', 'negatywny', 'neutralny']

    def process(self, dataframe: pd.DataFrame):
        dataframe['label'] = ""
        for i in range(len(dataframe)):
            if dataframe['ZNAK EMOCJI'][i] > 0:
                dataframe['label'][i] = 'pozytywny'
            elif dataframe['ZNAK EMOCJI'][i] < 0:
                dataframe['label'][i] = 'negatywny'
            else:
                dataframe['label'][i] = 'neutralny'

    def process_texts(self, dataframe: pd.DataFrame, texts: pd.DataFrame):
        dataframe['text'] = ""
        for i in range(len(dataframe)):
            x = texts.query('text_id == {}'.format(dataframe['text_id'][i]))
            dataframe['text'][i] = x['text'].values[0]


class CawiTwoAroDataset(CawiTwoDataset):
    def name(self):
        return "cawi2-aro"

    def label(self):
        return ['POBUDZENIE EMOCJONALNE', 'brak']

    def process(self, dataframe: pd.DataFrame):
        dataframe['label'] = ""
        for i in range(len(dataframe)):
            for lab in self.label():
                if lab is not 'brak' and dataframe[lab][i]:
                    dataframe['label'][i] = lab
            if dataframe['label'][i] not in self.label():
                dataframe['label'][i] = 'brak'


class CawiTwoEmoDataset(CawiTwoDataset):
    def name(self):
        return "cawi2-emo"

    def label(self):
        return ['RADOŚĆ','ZAUFANIE','OCZEKIWANIE','ZASKOCZENIE','STRACH','SMUTEK','WSTRĘT','ZŁOŚĆ', 'brak']

    def process(self, dataframe: pd.DataFrame):
        dataframe['label'] = ""
        for i in range(len(dataframe)):
            for lab in self.label():
                if lab is not 'brak' and dataframe[lab][i]:
                    dataframe['label'][i] = lab
            if dataframe['label'][i] not in self.label():
                dataframe['label'][i] = 'brak'

