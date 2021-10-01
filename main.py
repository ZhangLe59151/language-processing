# --*-- coding: utf-8 --*--
import pandas as pd
import emoji
from transformers import pipeline

def xlsx_to_csv():
    data_xlsx = pd.read_excel('./dataset.xlsx', engine='openpyxl', index_col=0)
    data_xlsx.to_csv('./dataset.csv', encoding='utf-8')

def clean_data():
    news = pd.read_table('./dataset.csv', header = None, name = ['content', 'datetime'])
    news['text'] = news['text'].apply(lambda x: emoji.demojize(str(x))) # translate the emoji to text
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    news['segment'] = news['text'].apply(lambda x: sentiment_analysis(str(x)[:512])[0]['label'])
    return news


if __name__ == '__main__':
    xlsx_to_csv()
    news = clean_data()
    # print(news)
