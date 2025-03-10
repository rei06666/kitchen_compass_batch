# KITCHEN_COMPASS_BATCH

## 概要
kitichen compass(レシピレコメンドアプリ)で使用する特徴量生成モデル、レシピデータベース作成用バッチ

## 前提条件
python 3.10  
GPU, CUDAが使える環境(CPUでも処理は可能)

## 利用手順
### レシピデータベース作成  
レシピデータベースを作成します。取得したデータに前処理をかけたうえで、レシピをベクトルデータに変換します  
データベースは作成後、kitchen compassアプリケーションに配置します
- requirements.txtを使用し、必要なパッケージをインストールします
```sh
pip install -r requirements.txt
```

- 楽天レシピ利用のために、楽天のアプリケーションIDを発行します。発行後、config.pyに記載します。
```python
class config: 
    def __init__(self):
        # 楽天レシピのアプリケーションID
        self.applicationid = "<ID>"
```
- レシピデータを取得しSQliteのデータベースに格納します  
```
python fetch_data.py
```
- recipes.dbが作成されていることを確認します
- データの前処理を行います。前処理データはデータベースに格納されます
```
python prepprocess_recipes.py
```
- データをベクトル化し、データベースに格納します。Sentence-Bertの学習済みモデルを使用しベクトル化しています
```
python create_vector.py
```
