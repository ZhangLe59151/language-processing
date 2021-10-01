# language-processing
customer reviews text from a product.  
extract the most liked part and disliked part from customer's reviews.    

- bert-segment-classification model  
- tokenization  
- POS tag  

## project strcuter：  
./data/dataset.xlsx  #There is a example file without data in the repo, please replace it to your dataset
./main.py 
./requirements.txt   

## project result:  
./data/negtive.csv
./data/positove.csv  

## porject docker  
20170327/nlpreviewstextprocessing  

## How to run  
docker pull 20170327/nlpreviewstextprocessing:latest  
docker run -it -v /this-repo-path:/usr/src/app 20170327/nlpreviewstextprocessing:latest /bin/bash  
cd /usr/src/app  
python3 main.py  