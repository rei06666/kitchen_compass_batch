from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import re
from db import Category, Recipe
import requests
from config import config
from time import sleep

conf = config()

# SQLiteのDB接続
DATABASE_URL = "sqlite:///recipes.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def fetch_and_store_categories():
    """楽天レシピAPIからカテゴリを取得し、DBに保存"""
    api_url = f"https://app.rakuten.co.jp/services/api/Recipe/CategoryList/20170426?applicationId={conf.applicationid}"
    res = requests.get(api_url)
    json_data = res.json()

    loop_count = 1

    for category_kind in json_data['result']:
        for category in json_data['result'][category_kind]:
            if "parentCategory" in category:
                parent_category = category['parentCategory']
            else:
                parent_category = None
            category_entry = Category(
                category=loop_count,
                category_id=category['categoryId'],
                category_name=category['categoryName'],
                category_url=category['categoryUrl'],
                parent_category=parent_category
            )
            session.add(category_entry)
            session.commit()
        loop_count += 1

    

def extract_category_id(recipe_url):
    """レシピURLからカテゴリIDを抽出"""
    match = re.search(r"/category/([\d-]+)/", recipe_url)
    return match.group(1) if match else None

def fetch_and_store_recipes():

    # DB内の全レシピURLを取得
    category_urls = session.query(Category.category_url).all()

    # # レシピURLからカテゴリIDを抽出
    category_ids = set()
    for url in category_urls:
        category_id = extract_category_id(url[0])
        if category_id:
            category_ids.add(category_id)

    for category_id in category_ids:
        api_url = f"https://app.rakuten.co.jp/services/api/Recipe/CategoryRanking/20170426?categoryId={category_id}&applicationId={conf.applicationid}"
        res = requests.get(api_url)
        sleep(1)
        json_data = res.json()

        for rcp in json_data.get('result', []):
            # レシピの存在確認
            existing_recipe = session.query(Recipe.recipe_id).filter(Recipe.recipe_id == rcp['recipeId']).first()

            if existing_recipe is None:
                recipe = Recipe(
                    recipe_id=rcp['recipeId'],
                    recipe_name=rcp['recipeTitle'],
                    recipe_url=rcp['recipeUrl'],
                    recipe_photo=rcp['foodImageUrl'],
                    material=", ".join(rcp['recipeMaterial']),
                    description=rcp.get('recipeDescription', ""),
                )
                session.add(recipe)

        session.commit()


if __name__ == "__main__":
    # fetch_and_store_categories()
    # fetch_and_store_recipes()
    print(conf.applicationid)
