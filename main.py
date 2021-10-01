# --*-- coding: utf-8 --*--
import pandas as pd
import emoji
from transformers import pipeline
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

def xlsx_to_csv():
    data_xlsx = pd.read_excel('./data/dataset.xlsx', engine='openpyxl', index_col=0)
    data_xlsx.to_csv('./data/dataset.csv', encoding='utf-8')

def clean_data():
    news = pd.read_csv('./data/dataset.csv', encoding='utf-8', names=['text','time'], header=0)
    news['text'] = news['text'].apply(lambda x: emoji.demojize(str(x))) # translate the emoji to text
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    news['segment'] = news['text'].apply(lambda x: sentiment_analysis(str(x)[:512])[0]['label'])
    return news

def processing_segment(df,label):
    LABEL_POSITIVE = df['segment']== label
    df = df[LABEL_POSITIVE]
    return df

def get_word_list(df):
    pos_pd = pd.DataFrame( columns = ['word', 'pos'])
    df['pos'] = df['text'].map(lambda x: pos_tag(word_tokenize(x)))
    word_List = df['pos'].values
    index = 0
    for item in word_List:
        for n in item:
            pos_pd.loc[index] = n
        index = index + 1
    nn_word = pos_pd['pos'].isin(['NN','NNP','NNS','NNPS','UNKNOWN'])
    Emoj_word_1 = pos_pd['word'].str.contains(':', regex=False)
    Emoj_word_2 = pos_pd['word'].str.contains('_', regex=False)
    nn_df = pos_pd[nn_word]
    nn_df = nn_df[~Emoj_word_1]
    nn_df = nn_df[~Emoj_word_2]
    return nn_df

def frequent_word(df):
    df['word'].values
    porter = nltk.PorterStemmer()
    tokens_porter=[ porter.stem(t) for t in df['word'].values ]
    stem_df = pd.DataFrame(tokens_porter, columns = ['word']) 
    return stem_df.value_counts()[:50]

if __name__ == '__main__':
    xlsx_to_csv()
    df = clean_data()
    positive_df = processing_segment(df, 'POSITIVE')
    negtive_df = processing_segment(df, 'NEGATIVE')
    positive_word_df = get_word_list(positive_df)
    negtive_word_df = get_word_list(negtive_df)
    positive_result_list = frequent_word(positive_word_df)
    negtive_result_list = frequent_word(negtive_word_df)
    positive_result_list.to_csv('./data/positive.csv')
    negtive_result_list.to_csv('./data/negtive.csv')