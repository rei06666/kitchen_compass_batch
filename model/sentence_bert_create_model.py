from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
BATCH_SIZE = 64
MAX_LENGTH = 64
NUM_EPOCHS = 10
USE_CUDA = True

np.random.seed(20221024)
torch.manual_seed(20221025)

ROOTDIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = ROOTDIR / 'data'  / 'train-v1.1.json'
VALID_DATA_PATH = ROOTDIR / 'data'  / 'valid-v1.1.json'


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_json(TRAIN_DATA_PATH, lines=True).set_index('sentence_pair_id')
    df_test = pd.read_json(VALID_DATA_PATH, lines=True).set_index('sentence_pair_id')

    # コサイン類似度で回帰するために、[0, 5] の値を [0, 1] にスケーリングする
    df_train['label'] = df_train['label'] / 5.0
    df_test['label'] = df_test['label'] / 5.0

    random_idx = np.random.permutation(len(df_train))
    n_train = int(len(df_train) * 0.8)
    df_train, df_valid = df_train.iloc[random_idx[:n_train]], df_train.iloc[random_idx[n_train:]]
    return df_train, df_valid, df_test


def encode_single_sentences(sentences: pd.Series, tokenizer: AutoTokenizer) -> list[dict[str, torch.Tensor]]:
    encoded_input = tokenizer(
        sentences.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return encoded_input


def encode_input(df: pd.DataFrame, tokenizer: AutoTokenizer) -> list[dict[str, torch.Tensor]]:
    encoded_input1 = encode_single_sentences(df['sentence1'], tokenizer)
    encoded_input2 = encode_single_sentences(df['sentence2'], tokenizer)
    dataset_for_loader = []
    for i in range(len(df)):
        d1 = {f'{k}_1': v[i] for k, v in encoded_input1.items()}
        d2 = {f'{k}_2': v[i] for k, v in encoded_input2.items()}
        d = d1 | d2 
        d['labels'] = torch.tensor(df.label.iloc[i], dtype=torch.float32)
        dataset_for_loader.append(d)
    return dataset_for_loader


def get_data_loaders(df_train, df_valid, df_test, tokenizer):
    train_encoded_input = encode_input(df_train, tokenizer)
    valid_encoded_input = encode_input(df_valid, tokenizer)
    test_encoded_input = encode_input(df_test, tokenizer)

    train_dataloader = DataLoader(train_encoded_input, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_encoded_input, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_encoded_input, batch_size=BATCH_SIZE)
    return train_dataloader, valid_dataloader, test_dataloader


class SentenceBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
    
    def forward(self, batch: dict[str, torch.Tensor]):
        # vector1
        batch1 = {k[:-2]: v for k, v in batch.items() if k.endswith('_1')}
        output1 = self.bert(**batch1)
        vector1 = self._mean_pooling(output1, batch1['attention_mask'])

        # vector2
        batch2 = {k[:-2]: v for k, v in batch.items() if k.endswith('_2')}
        output2 = self.bert(**batch2)
        vector2 = self._mean_pooling(output2, batch2['attention_mask'])

        cosine_similarity = torch.nn.functional.cosine_similarity(vector1, vector2)
        return cosine_similarity
    
    def _mean_pooling(self, bert_output, attention_mask):
        token_embeddings = bert_output.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(axis=1), min=1)

    
def train_step(model, batch, optimizer):
    model.train()
    if USE_CUDA:
        batch = {k: v.cuda() for k, v in batch.items()}
    cosine_similarity = model(batch)
    loss = torch.nn.functional.mse_loss(cosine_similarity, batch['labels'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    SUMMARY_WRITER.add_scalar('train/loss', loss.item(), GLOBAL_STEP)
    torch.cuda.empty_cache()
    return loss


def evaluate_model(model, dataloader):
    model.eval()
    y_test, y_pred = [], []
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            if USE_CUDA:
                batch = {k: v.cuda() for k, v in batch.items()}
            cosine_similarity = model(batch)

            # loss
            loss = torch.nn.functional.mse_loss(cosine_similarity, batch['labels'])
            total_loss += loss.item()

            # correlations
            y_test.append(batch['labels'].tolist())
            y_pred.append(cosine_similarity.tolist())

    y_test = sum(y_test, [])
    y_pred = sum(y_pred, [])

    loss = total_loss / len(dataloader)
    pearsonr = stats.pearsonr(y_test, y_pred)[0]
    spearmanr = stats.spearmanr(y_test, y_pred).correlation
    return loss, pearsonr, spearmanr


def validate_and_savemodel(model, dataloader, max_valid_spearmanr):
    loss, pearsonr, spearmanr = evaluate_model(model, dataloader)
    SUMMARY_WRITER.add_scalars(
        'valid',
        {'loss': loss, 'pearsonr': pearsonr, 'spearmanr': spearmanr},
        GLOBAL_STEP,
    )

    if spearmanr > max_valid_spearmanr:
        print(f'Saved model with spearmanr: {spearmanr:.4f} <- {max_valid_spearmanr:.4f}')
        torch.save(model.state_dict(), 'best_sentence_bert_model.bin')
    return loss, pearsonr, spearmanr


if __name__ == '__main__':
    # 開始
    start_time = datetime.now()
    SUMMARY_WRITER = SummaryWriter(log_dir='sentence_bert_experiment')

    # データとモデルのロード
    df_train, df_valid, df_test = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SentenceBert()
    if USE_CUDA:
        model = model.cuda()

    # モデルの学習
    max_valid_spearmanr = float('-inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(df_train, df_valid, df_test, tokenizer)
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_dataloader):
            GLOBAL_STEP = epoch * len(train_dataloader) + i
            _ = train_step(model, batch, optimizer)
            
            # validation step
            if i % 10 == 0:
                loss, pearsonr, spearmanr = validate_and_savemodel(model, valid_dataloader, max_valid_spearmanr)
                print(f'[epoch: {epoch}, iter: {i}] loss: {loss:.4f}, pearsonr: {pearsonr:.4f}, spearmanr: {spearmanr:.4f}')
                max_valid_spearmanr = max(max_valid_spearmanr, spearmanr)

    # モデルのテスト
    _, _, _ = validate_and_savemodel(model, valid_dataloader, max_valid_spearmanr)
    model.load_state_dict(torch.load('best_sentence_bert_model.bin'))
    loss, pearsonr, spearmanr = evaluate_model(model, test_dataloader)
    print(f'[Best model] loss: {loss:.4f}, pearsonr: {pearsonr:.4f}, spearmanr: {spearmanr:.4f}')
    # 終了
    SUMMARY_WRITER.close()
    end_time = datetime.now()
    print(f'実行時間: {(end_time - start_time).seconds} 秒')