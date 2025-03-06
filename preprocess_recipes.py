import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Recipe
import neologdn
import emoji

# SQLiteのDB接続
DATABASE_URL = "sqlite:///recipes.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def preprocess_text(text):
    # 記号や絵文字を取り除く正規表現パターン
    preprocessed_text = neologdn.normalize(text)
    preprocessed_text.rstrip('\n')
    preprocessed_text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', preprocessed_text)
    preprocessed_text = ''.join(['' if emoji.is_emoji(c) else c for c in preprocessed_text])
    preprocessed_text = re.sub(r'[!-/:-@[-`{-~]', r' ', preprocessed_text)
    # 全角記号の置換 (ここでは0x25A0 - 0x266Fのブロックのみを除去)
    preprocessed_text = re.sub(u'[■-♯]', ' ', preprocessed_text)
    preprocessed_text = re.sub(r'[【】]', ' ', preprocessed_text)
    return preprocessed_text

def preprocess_recipes():
    # DBから全てのレシピを取得
    recipes = session.query(Recipe).all()
    
    for recipe in recipes:
        if recipe.recipe_name and recipe.description:
            combined_text = f"{recipe.recipe_name} {recipe.description}"
            preprocessed_text = preprocess_text(combined_text)
            recipe.preprocessed_description = preprocessed_text
            session.add(recipe)
            print(f"Preprocessed: {recipe.recipe_name}")
    
    session.commit()
    print("Preprocessing and commit completed")

def export_preprocessed_descriptions(output_file):
    # DBから全てのレシピを取得
    recipes = session.query(Recipe).all()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for recipe in recipes:
            if recipe.preprocessed_description:
                f.write(f"{recipe.preprocessed_description}\n")
    print(f"Exported preprocessed descriptions to {output_file}")

if __name__ == '__main__':
    # preprocess_recipes()
    export_preprocessed_descriptions('preprocessed_descriptions.txt')