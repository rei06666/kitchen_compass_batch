from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.sentence_bert_create_model import SentenceBert, encode_single_sentences
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Recipe

USE_CUDA = False
BATCH_SIZE = 64
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# SQLiteのDB接続
DATABASE_URL = "sqlite:///recipes.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()


class SentenceBertSearcher:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sentence_bert = SentenceBert()
        self.sentence_bert.load_state_dict(torch.load('model/best_sentence_bert_model.bin', map_location=torch.device('cpu')))
        if USE_CUDA:
            self.sentence_bert = self.sentence_bert.cuda()
        self.sentence_bert.eval()
    
    def make_sentence_vectors(self, texts):
        encoded = encode_single_sentences(texts, self.tokenizer)
        dataset_for_loader = [
            {k: v[i] for k, v in encoded.items()}
            for i in range(len(texts))
        ]
        sentence_vectors = []
        for batch in DataLoader(dataset_for_loader, batch_size=BATCH_SIZE):
            if USE_CUDA:
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                bert_output = self.sentence_bert.bert(**batch)
                sentence_vector = self.sentence_bert._mean_pooling(bert_output, batch['attention_mask'])
                sentence_vectors.append(sentence_vector.cpu().detach().numpy())
        sentence_vectors = np.vstack(sentence_vectors)
        return sentence_vectors


def update_recipe_vectors():
    # DBから全てのレシピを取得
    recipes = session.query(Recipe).all()
    
    searcher = SentenceBertSearcher()
    # ベクトルを生成
    for i, recipe in enumerate(recipes):
        if recipe.description:
            vector = searcher.make_sentence_vectors(pd.Series([recipe.preprocessed_description]))[0]
            recipe.vector = vector.tobytes()
            session.add(recipe)

    session.commit()
    print("Final commit completed")


if __name__ == '__main__':
    # レシピのベクトルを更新
    update_recipe_vectors()