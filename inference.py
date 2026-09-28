"""Создание ответа для бенчмарка с помощью BM25, E5, географии и RRF."""
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import BallTree
from train import HERE, DATA, CACHE, prepare_embeddings

OUTPUT = HERE / 'answer.csv'
GEO_ARTIFACT = CACHE / 'geo_artifact.json'
ITEMS = ['item_id', 'item_title_raw', 'item_infm_params_text',
         'item_description_raw', 'item_location_id']
DEPTH = 100
OUTPUT_DEPTH = 50
RELATED_LOCATIONS = 10
RELATED_BUDGET = 10
RRF_K = 5
RADIUS_KM = 100
RADIUS_QUOTA = 20
EXACT_RADIUS_QUOTA = 10
EARTH_KM = 6371.0088


def normalize(text):
    return re.sub(r'[^\w]+', ' ', str(text).lower()).strip()


def bm25(counts, k1, b):
    #Взвешиваем разреженную матрицу документов по формуле BM25 с учётом IDF.
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    normalizer = k1 * ((1-b) + b*lengths/lengths.mean())
    frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((counts.shape[0]-frequency+.5)/(frequency+.5)).astype(np.float32)
    matrix = counts.tocsr(copy=True)
    row_normalizers = np.repeat(normalizer, np.diff(matrix.indptr))
    matrix.data = matrix.data * (k1+1)/(matrix.data+row_normalizers)
    matrix.data *= idf[matrix.indices]
    return matrix


def geographic_scopes(corpus, queries, mapping):
    #Собрать основной и дополнительный наборы объявлений для каждой локации.
    by_location = {loc: group.index.to_numpy()
                   for loc, group in corpus.groupby('item_location_id')}
    base, extra = {}, {}
    for loc in queries.search_location_id.unique():
        related = mapping.get(loc, ())
        original = by_location.get(loc)
        if original is None:
            arrays = [by_location[item_loc] for item_loc in related[:5] if item_loc in by_location]
            original = np.unique(np.concatenate(arrays)) if arrays else np.array([], np.int64)
        arrays = [by_location[item_loc] for item_loc in related[:RELATED_LOCATIONS]
                  if item_loc in by_location]
        extended = np.unique(np.concatenate(arrays)) if arrays else np.array([], np.int64)
        base[loc] = original
        extra[loc] = np.setdiff1d(extended, original, assume_unique=True)
    return base, extra


def word_candidates(index, qmatrix, locations, scope):
    output = np.full((len(locations), DEPTH), -1, np.int32)
    for loc in np.unique(locations):
        rows = np.flatnonzero(locations == loc)
        candidates = scope[loc]
        if not len(candidates):
            continue
        subindex = index[candidates].T.tocsr()
        for start in range(0, len(rows), 512):
            batch = rows[start:start+512]
            scores = (qmatrix[batch] @ subindex).tocsr()
            for j, row_idx in enumerate(batch):
                row = scores.getrow(j)
                count = min(DEPTH, row.nnz)
                if count:
                    best = np.argpartition(row.data, -count)[-count:]
                    best = best[np.argsort(-row.data[best], kind='stable')]
                    output[row_idx, :count] = candidates[row.indices[best]]
    return output


def budget_merge(base, extra):
    # Выделяем 10 из 100 мест кандидатам из связанных локаций.
    raw = np.concatenate((base, extra), axis=1)
    ranks = np.arange(1, DEPTH+1, dtype=np.float64)
    priority = np.tile(np.concatenate((ranks/(DEPTH-RELATED_BUDGET),
                                       ranks/RELATED_BUDGET)), (len(base), 1))
    priority[raw < 0] = np.inf
    order = np.argsort(priority, axis=1, kind='stable')[:, :DEPTH]
    output = np.take_along_axis(raw, order, axis=1).copy()
    output[~np.isfinite(np.take_along_axis(priority, order, axis=1))] = -1
    return output


def rrf_top50(word, dense):
    # Объединяем лексический и векторный списки, суммируя веса совпадений.
    output = np.full((len(word), OUTPUT_DEPTH), -1, np.int32)
    weights = np.tile(1/(RRF_K+np.arange(1, DEPTH+1, dtype=np.float64)), 2)
    for start in range(0, len(word), 4096):
        stop = min(start+4096, len(word))
        raw = np.concatenate((word[start:stop], dense[start:stop]), axis=1)
        order = np.argsort(raw, axis=1, kind='stable')
        items = np.take_along_axis(raw, order, axis=1)
        scores = weights[order].copy()
        # В каждом исходном списке объявления уникальны. После сортировки
        # совпадения из двух списков соседствуют: складываем их веса один раз.
        same = (items[:, 1:] == items[:, :-1]) & (items[:, 1:] >= 0)
        scores[:, :-1] += np.where(same, scores[:, 1:], 0.)
        scores[:, 1:][same] = -np.inf
        scores[items < 0] = -np.inf
        # При равных весах сохраняем приоритет исходного порядка кандидатов.
        ranked = np.lexsort((order, -scores), axis=1)[:, :OUTPUT_DEPTH]
        chosen = np.take_along_axis(items, ranked, axis=1)
        chosen[np.take_along_axis(scores, ranked, axis=1) == -np.inf] = -1
        output[start:stop] = chosen
    return output


def make_centers(corpus, click_centers, locations):
    catalog = corpus.groupby('item_location_id')[['item_latitude', 'item_longitude']].median()
    centers = {}
    for loc in np.unique(locations):
        if loc in catalog.index:
            row = catalog.loc[loc]
            lat, lon = float(row.item_latitude), float(row.item_longitude)
        elif int(loc) in click_centers:
            lat, lon = click_centers[int(loc)]
        else:
            continue
        if np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            centers[loc] = (lat, lon)
    return centers


def radius_scopes(tree, point_ids, centers, locations, radius_km, exact_indices):
    unique_locs = np.unique(locations)
    points = np.deg2rad(np.array([centers[loc] for loc in unique_locs]))
    nearby = tree.query_radius(points, r=radius_km/EARTH_KM)
    return {loc: np.union1d(point_ids[candidates], exact_indices.get(loc, ())).astype(np.int32)
            for loc, candidates in zip(unique_locs, nearby)}


def merge_unique(base, extra, quota):
    # Смешиваем два списка по 100 кандидатов с заданной квотой для радиуса.
    result = np.full_like(base, -1)
    ranks = np.arange(1, DEPTH + 1)
    priorities = np.concatenate((ranks / (DEPTH - quota), ranks / quota))
    order = np.argsort(priorities, kind='stable')
    for row in range(len(base)):
        candidates = np.concatenate((base[row], extra[row]))
        chosen, seen = [], set()
        for position in order:
            item = int(candidates[position])
            if item >= 0 and item not in seen:
                chosen.append(item)
                seen.add(item)
                if len(chosen) == DEPTH:
                    break
        result[row, :len(chosen)] = chosen
    return result


@torch.inference_mode()
def dense_candidates(item_vectors, query_vectors, locations, scope, device):
    output = np.full((len(query_vectors), DEPTH), -1, np.int32)
    for loc in np.unique(locations):
        rows = np.flatnonzero(locations == loc)
        candidates = scope[loc]
        if len(candidates) == 0:
            continue
        documents = torch.as_tensor(
            np.asarray(item_vectors[candidates]).copy(), device=device
        ).T.contiguous()
        for start in range(0, len(rows), 64):
            batch = rows[start:start+64]
            query_batch = torch.as_tensor(np.asarray(query_vectors[batch]).copy(), device=device)
            top = (query_batch @ documents).topk(
                min(DEPTH, len(candidates)), dim=1
            ).indices.cpu().numpy()
            output[batch, :top.shape[1]] = candidates[top]
    return output


def load_inputs():
    # Загружаем корпус, запросы и обученные географические связи.
    with GEO_ARTIFACT.open(encoding='utf-8') as stream:
        artifact = json.load(stream)
    mapping = {int(loc): np.asarray(values, dtype=np.int64)
               for loc, values in artifact['related_item_locations'].items()}
    click_centers = {int(loc): tuple(values)
                     for loc, values in artifact['fallback_click_centers'].items()}
    queries = pd.read_parquet(DATA/'benchmark_queries.parquet')
    items = pd.read_parquet(DATA/'benchmark_items.parquet', columns=ITEMS+['item_latitude','item_longitude'])
    assert items.item_id.is_unique and queries.query_id.is_unique
    base, extra = geographic_scopes(items, queries, mapping)
    return items, queries, base, extra, click_centers


def retrieve_bm25(items, queries, base, extra):
    # Cтроим индекс BM25 и выбираем до 100 лексических кандидатов.
    titles = items.item_title_raw.fillna('').map(normalize)
    parameters = items.item_infm_params_text.fillna('').str.slice(stop=600).map(normalize)
    descriptions = items.item_description_raw.fillna('').str.slice(stop=600).map(normalize)
    # Повтор заголовка увеличивает его вклад в лексический поиск.
    documents = titles + ' ' + titles + ' ' + parameters + ' ' + descriptions
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=3, dtype=np.float32)
    counts = vectorizer.fit_transform(documents).tocsr()
    index = bm25(counts, k1=1.2, b=0.0)
    query_texts = queries.search_query.fillna('').map(normalize)
    query_filters = queries.search_infm_params_text.fillna('').str.slice(stop=300).map(normalize)
    query_matrix = vectorizer.transform(query_texts + ' ' + query_filters).tocsr()
    query_matrix.data[:] = 1
    locations = queries.search_location_id.to_numpy()
    print('Поиск кандидатов BM25', flush=True)
    word = budget_merge(
        word_candidates(index, query_matrix, locations, base),
        word_candidates(index, query_matrix, locations, extra),
    )
    return index, query_matrix, word


def retrieve_dense(items, queries, base, extra, device):
    #Выбираем до 100 кандидатов по готовым эмбеддингам E5.
    item_vectors, query_vectors = prepare_embeddings(items, queries, CACHE, device)
    locations = queries.search_location_id.to_numpy()
    print('Поиск кандидатов по эмбеддингам', flush=True)
    dense = budget_merge(
        dense_candidates(item_vectors, query_vectors, locations, base, device),
        dense_candidates(item_vectors, query_vectors, locations, extra, device),
    )
    return item_vectors, query_vectors, dense


def expand_with_radius(items, queries, click_centers, index, query_matrix, item_vectors,
                       query_vectors, word, dense, device):
    # Дополняем массивы word и dense на месте кандидатами из радиуса.
    # Для отсутствующих в каталоге локаций выделяем места кандидатам в радиусе.
    # Если точная локация есть, квота для радиуса меньше.
    locations = queries.search_location_id.to_numpy()
    no_exact = ~queries.search_location_id.isin(items.item_location_id).to_numpy()
    centers = make_centers(items, click_centers, locations)
    active = np.flatnonzero([loc in centers for loc in locations])
    if not len(active):
        return
    coordinates = items[['item_latitude','item_longitude']].to_numpy(dtype=np.float64)
    valid = (np.isfinite(coordinates).all(axis=1)
             & (np.abs(coordinates[:,0]) <= 90)
             & (np.abs(coordinates[:,1]) <= 180))
    point_ids = np.flatnonzero(valid).astype(np.int32)
    tree = BallTree(np.deg2rad(coordinates[valid]), metric='haversine', leaf_size=64)
    exact_indices = {loc: group.index.to_numpy(dtype=np.int32)
                     for loc, group in items.groupby('item_location_id')}
    radius_scope = radius_scopes(tree, point_ids, centers, locations[active], RADIUS_KM, exact_indices)
    print(f'Поиск в радиусе {RADIUS_KM} км: центры найдены для {len(active):,} запросов; '
          f'без точной локации: {no_exact.sum():,}', flush=True)
    radius_word = word_candidates(index, query_matrix[active], locations[active], radius_scope)
    radius_dense = dense_candidates(
        item_vectors, query_vectors[active], locations[active], radius_scope, device
    )
    groups = ((no_exact[active], RADIUS_QUOTA), (~no_exact[active], EXACT_RADIUS_QUOTA))
    for mask, quota in groups:
        radius_rows = np.flatnonzero(mask)
        if not len(radius_rows):
            continue
        query_rows = active[radius_rows]
        word[query_rows] = merge_unique(word[query_rows], radius_word[radius_rows], quota)
        dense[query_rows] = merge_unique(dense[query_rows], radius_dense[radius_rows], quota)


def save_answer(items, queries, word, dense, started):
    # Объединяем списки RRF и записываем CSV.
    ranked = rrf_top50(word,dense)
    ids = items.item_id.to_numpy()
    answer_ids = [ids[row[row >= 0]].tolist() for row in ranked]
    valid_ids = set(ids)
    for selected_ids in answer_ids:
        assert len(selected_ids) <= OUTPUT_DEPTH
        assert len(selected_ids) == len(set(selected_ids))
        assert set(selected_ids).issubset(valid_ids)
    result = pd.DataFrame({
        'query_id': queries.query_id,
        'answer': [' '.join(selected_ids) for selected_ids in answer_ids],
    })
    assert len(result) == len(queries) and result.query_id.is_unique
    result.to_csv(OUTPUT, index=False, encoding='utf-8')
    print(f'Сохранён {OUTPUT}; строк: {len(result):,}; '
          f'время: {time.monotonic()-started:.0f} с', flush=True)


def main():
    started = time.monotonic()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cpu':
        print('На CPU поиск может занять больше времени. =(', flush=True)
    torch.set_num_threads(8)
    items, queries, base, extra, click_centers = load_inputs()
    print(f'Объявлений: {len(items):,}; запросов: {len(queries):,}; '
          f'устройство для поиска: {device}', flush=True)
    index, query_matrix, word = retrieve_bm25(items, queries, base, extra)
    item_vectors, query_vectors, dense = retrieve_dense(items, queries, base, extra, device)
    expand_with_radius(items, queries, click_centers, index, query_matrix, item_vectors,
                       query_vectors, word, dense, device)
    save_answer(items, queries, word, dense, started)


if __name__ == '__main__':
    main()
