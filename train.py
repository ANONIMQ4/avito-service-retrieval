import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
CACHE = HERE / 'cache'
OUTPUT = CACHE / 'geo_artifact.json'
MODEL = 'intfloat/multilingual-e5-small'
SEARCH = ['search_query', 'search_location_id', 'search_is_delivery_search',
          'search_infm_params_text', 'search_category']
ITEM_TEXT_SIGNATURE = '9cf5c8a705780b33'
QUERY_TEXT_SIGNATURE = '699e465b905e6ed0'


def signature(texts):
    digest = hashlib.sha256(b'intfloat/multilingual-e5-small:max_length=128')
    for value in texts:
        digest.update(value.encode())
        digest.update(b'\0')
    return digest.hexdigest()[:16]


def cache_is_complete(path, rows):
    state = path.with_suffix('.json')
    if not path.exists() or not state.exists():
        return False
    if json.loads(state.read_text()).get('completed') != rows:
        return False
    vectors = np.load(path, mmap_mode='r')
    return vectors.shape == (rows, 384) and vectors.dtype == np.float32


@torch.inference_mode()
def encode(texts, path, model, tokenizer, device, batch_size=128):
    # Кодируем тексты пакетами с возможностью продолжить прерванный расчёт.
    path.parent.mkdir(parents=True, exist_ok=True)
    state = path.with_suffix('.json')
    done = 0
    if path.exists():
        vectors = np.load(path, mmap_mode='r+')
        assert vectors.shape == (len(texts), model.config.hidden_size)
        if state.exists():
            done = json.loads(state.read_text())['completed']
    else:
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                            shape=(len(texts), model.config.hidden_size))
    started = time.monotonic()
    for start in range(done, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        tokens = tokenizer(texts[start:stop], padding=True, truncation=True,
                           max_length=128, return_tensors='pt').to(device)
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        vectors[start:stop] = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        if stop == len(texts) or (stop // batch_size) % 100 == 0:
            vectors.flush()
            state.write_text(json.dumps({'completed': stop}))
            print(f'{path.name}: {stop:,}/{len(texts):,}; '
                  f'{time.monotonic()-started:.0f}s', flush=True)
    return vectors


def prepare_embeddings(items, queries, cache_dir=CACHE, device=None):
    # Читаем готовые векторы или пересчитываем их с помощью E5.
    item_texts = ('passage: ' + items.item_title_raw.fillna('') + '. '
                  + items.item_infm_params_text.fillna('').str.slice(stop=300) + '. '
                  + items.item_description_raw.fillna('').str.slice(stop=500)).tolist()
    query_filters = queries.search_infm_params_text.fillna('').str.lower().str.strip()
    query_suffixes = query_filters.map(lambda value: '. ' + value if value else '')
    query_texts = ('query: ' + queries.search_query.fillna('').str.lower().str.strip()
                   + query_suffixes).tolist()
    assert signature(item_texts) == ITEM_TEXT_SIGNATURE, 'Тексты объявлений не соответствуют кэшу'
    assert signature(query_texts) == QUERY_TEXT_SIGNATURE, 'Тексты запросов не соответствуют кэшу'
    item_cache = cache_dir / 'item_embeddings.npy'
    query_cache = cache_dir / 'query_embeddings.npy'
    missing = [(texts, path) for texts, path in
               ((item_texts, item_cache), (query_texts, query_cache))
               if not cache_is_complete(path, len(texts))]
    if missing:
        # Transformers нужен только при отсутствии готового кэша.
        from transformers import AutoModel, AutoTokenizer
        device = device or ('cuda' if torch.cuda.is_available() else
                            ('mps' if torch.backends.mps.is_available() else 'cpu'))
        print(f'Устройство для расчёта эмбеддингов: {device}', flush=True)
        # Готовые веса берутся из локального кэша; недостающие скачиваются.
        print(f'Загрузка модели {MODEL}: при необходимости файлы будут скачаны '
              'с Hugging Face.', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(device).eval()
        model.requires_grad_(False)
        for texts, path in missing:
            encode(texts, path, model, tokenizer, device)
    return np.load(item_cache, mmap_mode='r'), np.load(query_cache, mmap_mode='r')


def load_training_split():
    columns = SEARCH + ['item_id', 'item_location_id', 'item_latitude', 'item_longitude']
    source = pd.read_parquet(DATA / 'train.parquet', columns=columns)

    # Воспроизводим разбиение 70/30 по запросам. 
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(source[SEARCH].fillna('')), sort=False)
    selected = np.random.default_rng(42).random(codes.max() + 1) < .70
    return source.loc[selected[codes]]


def fit_geography(fit):
    # Считаем уникальные пары «запрос — объявление» и упорядочиваем локации
    # объявлений по частоте для каждой поисковой локации. Используем до десяти.
    counts = fit.drop_duplicates(SEARCH + ['item_id']).groupby(
        ['search_location_id', 'item_location_id']).size()
    related = {
        str(int(loc)): [int(value) for value in values.sort_values(ascending=False)
                        .index.get_level_values(1)]
        for loc, values in counts.groupby(level=0)
    }

    # Если в каталоге нет объявлений точной локации, оцениваем её центр
    # по координатам выбранных объявлений из обучающей части.
    medians = fit.groupby('search_location_id')[['item_latitude', 'item_longitude']].median()
    click_centers = {
        str(int(loc)): [float(row.item_latitude), float(row.item_longitude)]
        for loc, row in medians.iterrows()
    }
    return {
        'split_seed': 42,
        'train_share': .70,
        'fit_rows': len(fit),
        'related_item_locations': related,
        'fallback_click_centers': click_centers,
    }


def main():
    artifact = fit_geography(load_training_split())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, separators=(',', ':')),
                      encoding='utf-8')
    print(f'Сохранён {OUTPUT}: обучающих пар — {artifact["fit_rows"]:,}, '
          f'поисковых локаций — {len(artifact["related_item_locations"]):,}')
    # Записи бенчмарка не размечены: здесь они нужны только для подготовки
    # векторов текста, которые затем использует inference.py.
    items = pd.read_parquet(DATA / 'benchmark_items.parquet', columns=[
        'item_title_raw', 'item_infm_params_text', 'item_description_raw'])
    queries = pd.read_parquet(DATA / 'benchmark_queries.parquet', columns=[
        'search_query', 'search_infm_params_text'])
    prepare_embeddings(items, queries, CACHE)
    print('Кэш эмбеддингов готов')


if __name__ == '__main__':
    main()
