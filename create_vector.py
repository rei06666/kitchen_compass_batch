import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Recipe

MODEL_NAME = 'sonoisa/sentence-bert-base-ja-mean-tokens'

# SQLiteのDB接続
DATABASE_URL = "sqlite:///recipes.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# Sentence-BERTのモデルを使ってレシピのベクトルを生成
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="max_length", max_length=512,
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


def update_recipe_vectors():
    # DBから全てのレシピを取得
    recipes = session.query(Recipe).all()
    
    # searcher = SentenceBertSearcher()
    sentenceBert= SentenceBertJapanese(MODEL_NAME)
    # ベクトルを生成
    for i, recipe in enumerate(recipes):
        if recipe.description:
            vector = sentenceBert.encode(pd.Series([recipe.preprocessed_description]))[0]
            print("encoded")
            recipe.vector = vector
            session.add(recipe)

    session.commit()
    print("Final commit completed")


if __name__ == '__main__':
    # レシピのベクトルを更新
    update_recipe_vectors()