import numpy as np
import polars as pl
import scipy.sparse as sp
import json
import ast
import pickle
import gc
import re

from rank_bm25 import BM25Okapi
from rouge import Rouge
from rapidfuzz import fuzz
import jellyfish
import textdistance

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
tqdm.pandas()

df = pl.read_parquet('../data/preprocessed/all_products_preprocessed.parquet')

print(f'{df.shape=}')

gc.collect()

# коссим по тфидф векторам

def add_cosine_similarity_from_files(
    train_df: pl.DataFrame,
    train_tpl_1: str,
    train_tpl_2: str,
    cols: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    for col in cols:
        print(f'{col=}')
        
        print('loading tr1')
        tr1 = sp.load_npz(train_tpl_1.format(col=col))
        
        print('loading tr2')
        tr2 = sp.load_npz(train_tpl_2.format(col=col))
        

        train_sims = []
        for i in tqdm(range(tr1.shape[0]), desc=f'Train rows for {col}', leave=False):
            if tr1.getrow(i).nnz or tr2.getrow(i).nnz:
                sim = float(cosine_similarity(tr1.getrow(i), tr2.getrow(i))[0, 0])
            else:
                sim = 0.0
            train_sims.append(sim)

        sim_col = f"{col}_cosine_sim"
        train_df = train_df.with_columns(pl.Series(sim_col, train_sims))

        del train_sims
        gc.collect()

    return train_df

df = add_cosine_similarity_from_files(
    df,
    '../data/preprocessed/{col}_1_tfidf_matrices.npz',
    '../data/preprocessed/{col}_2_tfidf_matrices.npz',
    ['concat_keyval', 'name', 'description']
)

# мэтч по категориям 1-4 уровня + частичный мэтч по 4 уровню

for i in range(1, 5):
    df = df.with_columns(
        pl.col(f'category_level_{i}_1').cast(pl.String).str.to_lowercase().eq(
            pl.col(f'category_level_{i}_2').cast(pl.String).str.to_lowercase()
        ).alias(f'category_level_{i}_match')
    )
    
    if i == 4:
        def calc_token_sort_ratio(row):
            val1 = str(row[f'category_level_4_1'])
            val2 = str(row[f'category_level_4_2'])
            return fuzz.token_sort_ratio(val1, val2) / 100
        
        total = len(df)
        pbar = tqdm(total=total, desc=f'Processing token_sort_ratio for level {i}')
        
        def token_sort_with_progress(row):
            pbar.update(1)
            return calc_token_sort_ratio(row)
        
        df = df.with_columns(
            pl.struct([f'category_level_4_1', f'category_level_4_2'])
            .map_elements(token_sort_with_progress, return_dtype=pl.Float64)
            .alias(f'category_level_4_token_sort_ratio_match')
        )
        pbar.close()

gc.collect()

# всего совпадений по категориям

df = df.with_columns(
    (pl.col('category_level_1_match').fill_null(False).cast(pl.Int64) + 
        pl.col('category_level_2_match').fill_null(False).cast(pl.Int64) + 
        pl.col('category_level_3_match').fill_null(False).cast(pl.Int64) + 
        pl.col('category_level_4_match').fill_null(False).cast(pl.Int64))
    .alias('category_total_matches')
)
    
gc.collect()

# отношения длин для всего подряд (левые и правые)

num_cols = [
    'price', 'n_images', 
    'name_tokens_len', 'description_tokens_len', 
    'name_en_tokens_len', 'description_en_tokens_len',
    'name_mix_tokens_len', 'description_mix_tokens_len',
    'attr_keys_len', 'attr_vals_len',
]

str_cols = ['name_tokens_w_digits', 'description_tokens_w_digits']

for col in num_cols:
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} ratios")

    def ratio_left(row):
        pbar.update(1)
        val2 = row[f'{col}_2']
        val1 = row[f'{col}_1']
        
        if val2 is None or val2 == 0 or val1 is None:
            return None
        
        return float(val1) / float(val2)

    def ratio_right(row):
        val2 = row[f'{col}_2']
        val1 = row[f'{col}_1']
        
        if val1 is None or val1 == 0 or val2 is None:
            return None
            
        return float(val2) / float(val1)

    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            ratio_left, return_dtype=pl.Float64
        ).alias(f'{col}_ratio_left'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            ratio_right, return_dtype=pl.Float64
        ).alias(f'{col}_ratio_right')
    )
    pbar.close()

for col in str_cols:
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} ratios")

    def str_ratio_left(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        
        if val1 is None:
            len1 = 0
        else:
            len1 = len(str(val1).split())
            
        if val2 is None:
            len2 = 0
        else:
            len2 = len(str(val2).split())
            
        if len2 == 0:
            return 0.0
            
        return float(len1) / float(len2)

    def str_ratio_right(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        
        if val1 is None:
            len1 = 0
        else:
            len1 = len(str(val1).split())
            
        if val2 is None:
            len2 = 0
        else:
            len2 = len(str(val2).split())
            
        if len1 == 0:
            return 0.0
            
        return float(len2) / float(len1)

    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            str_ratio_left, return_dtype=pl.Float64
        ).alias(f'{col}_ratio_left'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            str_ratio_right, return_dtype=pl.Float64
        ).alias(f'{col}_ratio_right')
    )
    pbar.close()
    
gc.collect()

# разницы длин

def abs_len_diff_features(df, cols):
    for col in cols:
        total = len(df)
        pbar = tqdm(total=total, desc=f"Processing abs_diff for {col}")

        def abs_len_diff(row):
            pbar.update(1)
            val1 = row[f'{col}_1']
            val2 = row[f'{col}_2']
            if val1 is None or val2 is None:
                return None
            v1 = int(val1)
            v2 = int(val2)
            return abs(v1 - v2)

        df = df.with_columns(
            pl.struct([f'{col}_1', f'{col}_2'])
            .map_elements(abs_len_diff, return_dtype=pl.Int64)
            .alias(f'{col}_abs_diff')
        )
        pbar.close()
        
    return df

def abs_len_sqdiff_features(df, cols):
    for col in cols:
        total = len(df)
        pbar = tqdm(total=total, desc=f"Processing sq_diff for {col}")

        def sq_len_diff(row):
            pbar.update(1)
            val1 = row[f'{col}_1']
            val2 = row[f'{col}_2']
            if val1 is None or val2 is None:
                return None
            v1 = int(val1)
            v2 = int(val2)
            
            diff_sq = (v1 - v2) ** 2
            
            max_uint64 = 2 ** 64 - 1
            if diff_sq >= max_uint64:
                return max_uint64 - 1
            
            return diff_sq

        df = df.with_columns(
            pl.struct([f'{col}_1', f'{col}_2'])
            .map_elements(sq_len_diff, return_dtype=pl.UInt64)
            .alias(f'{col}_sq_diff')
        )
        pbar.close()
        
    return df

diff_cols = ['price', 'n_images', 'attr_keys_len', 'attr_vals_len']

df = abs_len_diff_features(df, diff_cols)
df = abs_len_sqdiff_features(df, diff_cols)
    
gc.collect()

# полнота некоторых столбцов с пропусками

def fillness(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    condition_both = (pl.col(f'{col_name}_1').is_not_null() & 
                      pl.col(f'{col_name}_2').is_not_null())
    condition_none = (pl.col(f'{col_name}_1').is_null() & 
                      pl.col(f'{col_name}_2').is_null())
    
    df = df.with_columns(
        pl.when(condition_both).then(pl.lit('both'))
        .when(condition_none).then(pl.lit('none'))
        .otherwise(pl.lit('only one'))
        .alias(f'{col_name}_fillness')
    )
    
    return df

df = fillness(df, 'category_level_3')
df = fillness(df, 'category_level_4')
df = fillness(df, 'n_images')
    
gc.collect()

# фичи на антисловах

with open('../data/preprocessed/filtered_anti_words.pkl', 'rb') as file:
    filtered_anti_words = pickle.load(file)

N_TOP_ANTIWORD = 100
top_anti_words = set([w[0] for w in filtered_anti_words.most_common(N_TOP_ANTIWORD)])

def calc_anti_words_values(row: dict) -> float:
    name1, name2 = row['name_1'], row['name_2']
    words1 = set(re.findall(r'([a-z]+)', name1.lower()))
    words2 = set(re.findall(r'([a-z]+)', name2.lower()))
    
    if not (words1 | words2):
        return 0.0
        
    xor_words = words1.symmetric_difference(words2)
    intersection = xor_words & top_anti_words
    return len(intersection) / max(len(words1), len(words2))

df = df.with_columns(
    pl.struct(['name_1', 'name_2'])
    .map_elements(calc_anti_words_values, return_dtype=pl.Float64)
    .alias('anti_words_values')
)
    
gc.collect()

# совпадения для словаря атрибутов

def avg_fully_eq_attributes(d1, d2):
    if d1 is None or d2 is None:
        return None
    try:
        d1 = ast.literal_eval(d1)
        d2 = ast.literal_eval(d2)
    except Exception:
        print('bad!!!')
        return None
    keys = set(d1) & set(d2)
    metrics = []
    for key in keys:
        metrics.append(set(d1[key]) == set(d2[key]))
    if len(metrics) == 0:
        return None
    return np.mean(metrics)

total = len(df)
pbar = tqdm(total=total, desc='Calculating avg_fully_eq_attributes')

def apply_avg_fully_eq_attributes(row):
    pbar.update(1)
    return avg_fully_eq_attributes(
        row['characteristic_attributes_mapping_1'], 
        row['characteristic_attributes_mapping_2']
    )

df = df.with_columns(
    pl.struct(['characteristic_attributes_mapping_1', 'characteristic_attributes_mapping_2'])
    .map_elements(apply_avg_fully_eq_attributes, return_dtype=pl.Float64)
    .alias('attributes_values_avg_fully_eq')
)
pbar.close()
    
gc.collect()

# частичные мэтчи по названиям и описаниям

def apply_string_metrics(df, col, include_extra_metrics=True):
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} similarity metrics")
    def token_sort_ratio(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return fuzz.token_sort_ratio(str(val1), str(val2)) / 100
    
    def token_set_ratio(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return fuzz.token_set_ratio(str(val1), str(val2)) / 100
    
    def jaro_winkler_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return jellyfish.jaro_winkler_similarity(str(val1), str(val2))
    
    def dice_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return textdistance.dice(str(val1), str(val2))
    
    def tanimoto_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return textdistance.tanimoto(str(val1), str(val2))
    
    def sorensen_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return textdistance.sorensen(str(val1), str(val2))
    
    new_columns = [
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            token_sort_ratio, return_dtype=pl.Float64
        ).alias(f'{col}_token_sort_ratio'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            token_set_ratio, return_dtype=pl.Float64
        ).alias(f'{col}_token_set_ratio'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            jaro_winkler_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_jaro_winkler_similarity'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            dice_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_dice'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            tanimoto_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_tanimoto'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            sorensen_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_sorensen')
    ]
    
    if include_extra_metrics and 'attr' not in col:
        def damerau_levenshtein_distance(row):
            val1 = row[f'{col}_1']
            val2 = row[f'{col}_2']
            if val1 is None or val2 is None:
                return None
            return jellyfish.damerau_levenshtein_distance(str(val1), str(val2))
        
        def wratio_similarity(row):
            val1 = row[f'{col}_1']
            val2 = row[f'{col}_2']
            if val1 is None or val2 is None:
                return None
            return fuzz.WRatio(str(val1), str(val2)) / 100
        
        new_columns.extend([
            pl.struct([f'{col}_1', f'{col}_2']).map_elements(
                damerau_levenshtein_distance, return_dtype=pl.Int64
            ).alias(f'{col}_damerau_levenshtein_distance'),
            
            pl.struct([f'{col}_1', f'{col}_2']).map_elements(
                wratio_similarity, return_dtype=pl.Float64
            ).alias(f'{col}_WRatio')
        ])
    
    df = df.with_columns(new_columns)
    pbar.close()
    return df

def apply_string_metrics_desc(df, col):
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} similarity metrics")
    def token_sort_ratio(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return fuzz.token_sort_ratio(str(val1), str(val2)) / 100
    
    def token_set_ratio(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return fuzz.token_set_ratio(str(val1), str(val2)) / 100
    
    def jaro_winkler_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return jellyfish.jaro_winkler_similarity(str(val1), str(val2))
    
    def dice_similarity(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        return textdistance.dice(str(val1), str(val2))
    
    new_columns = [
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            token_sort_ratio, return_dtype=pl.Float64
        ).alias(f'{col}_token_sort_ratio'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            token_set_ratio, return_dtype=pl.Float64
        ).alias(f'{col}_token_set_ratio'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            jaro_winkler_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_jaro_winkler_similarity'),
        
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(
            dice_similarity, return_dtype=pl.Float64
        ).alias(f'{col}_dice'),
    ]
    
    df = df.with_columns(new_columns)
    pbar.close()
    return df

for col in ('name', 'name_norm', 'name_en', 'name_mix', 'description_en', 'description_mix', 'name_tokens_w_digits', 'description_tokens_w_digits'):
    df = apply_string_metrics(df, col)
for col in ('description', 'description_norm'):
    df = apply_string_metrics_desc(df, col)
    
gc.collect()

# совпадения топ-атрибутов по категориям

pop_characts_tf_idf = {}
for level in range(1, 5):
    with open(f'../data/preprocessed/pop_characts_tf_idf_level_{level}.pkl', 'rb') as file:
        pop_characts_tf_idf[level] = pickle.load(file)

def generate_top_attribute_matches(df, level):
    if level == 1:
        TOP_N_characts = 75
    elif level == 2:
        TOP_N_characts = 50
    elif level == 3:
        TOP_N_characts = 50
    elif level == 4:
        TOP_N_characts = 25
    else:
        print('u are gay')

    total = len(df)
    pbar = tqdm(total=total, desc=f"Generating top attribute matches for level {level}")
    
    def code_top_tf_idf_characteristics(row):
        pbar.update(1)
        cat_1 = row[f'category_level_{level}_1']
        cat_2 = row[f'category_level_{level}_2']
        attr_1 = row['characteristic_attributes_mapping_1']
        attr_2 = row['characteristic_attributes_mapping_2']
        
        result = [0.0] * TOP_N_characts
        
        if attr_1 is None or attr_2 is None or cat_1 != cat_2 or cat_1 not in pop_characts_tf_idf[level]:
            return result
        
        try:
            attr_1 = json.loads(attr_1)
            attr_2 = json.loads(attr_2)
        except (json.JSONDecodeError, TypeError):
            print('bad!!!')
            return result
        
        top_pop_characts = [i[1] for i in pop_characts_tf_idf[level][cat_1][:TOP_N_characts]]
        
        for i, cat_name in enumerate(top_pop_characts):
            if i >= TOP_N_characts:
                break
            if cat_name in attr_1 and cat_name in attr_2:
                if attr_1[cat_name] == attr_2[cat_name]:
                    result[i] = 1.0
        
        return result
    
    df = df.with_columns(
        pl.struct([
            f'category_level_{level}_1', 
            f'category_level_{level}_2', 
            'characteristic_attributes_mapping_1', 
            'characteristic_attributes_mapping_2'
        ]).map_elements(code_top_tf_idf_characteristics, return_dtype=pl.List(pl.Float64))
        .alias(f'top_attr_match_list_lvl{level}')
    )
    
    new_cols = []
    new_col_names = []
    for i in range(TOP_N_characts):
        col_name = f'top_{i}_attr_match_lvl{level}'
        new_cols.append(
            pl.col(f'top_attr_match_list_lvl{level}').list.get(i).alias(col_name)
        )
        new_col_names.append(col_name)
    
    df = df.with_columns(new_cols)
    
    df = df.with_columns(
        pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in new_col_names])
        .alias(f'total_top_attribute_matches_lvl{level}')
    )
    
    df = df.drop(f'top_attr_match_list_lvl{level}')
    pbar.close()
    
    return df

for level in range(1, 5):
    df = generate_top_attribute_matches(df, level)
    gc.collect() 

gc.collect()

# взвешенное совпадение атрибутов

def generate_weighted_attribute_matches(df, level):
    if level == 1:
        n_values = [10, 25, 50, 75, 100, 150, 200, 250, 500]
    elif level == 2:
        n_values = [10, 25, 50, 75, 100, 150, 200]
    elif level == 3:
        n_values = [10, 25, 50, 75, 100]
    elif level == 4:
        n_values = [10, 25, 50]
    else:
        print('u are gay')
    
    total = len(df)
    pbar = tqdm(total=total, desc=f"Generating weighted attribute matches for level {level}")
    
    def code_weighted_tf_idf_characteristics(row):
        pbar.update(1)
        cat_1 = row[f'category_level_{level}_1']
        cat_2 = row[f'category_level_{level}_2']
        attr_1 = row['characteristic_attributes_mapping_1']
        attr_2 = row['characteristic_attributes_mapping_2']
        
        result = {n: 0.0 for n in n_values}
        
        if attr_1 is None or attr_2 is None or cat_1 != cat_2 or cat_1 not in pop_characts_tf_idf[level]:
            return [result[n] for n in n_values]
        
        try:
            attr_1 = json.loads(attr_1)
            attr_2 = json.loads(attr_2)
        except (json.JSONDecodeError, TypeError):
            return [result[n] for n in n_values]
        
        top_pop_characts = pop_characts_tf_idf[level][cat_1]
        
        for n in n_values:
            limit = min(n, len(top_pop_characts))
            weighted_sum = 0.0
            total_weight = 0.0
            for i in range(limit):
                weight, cat_name = top_pop_characts[i]
                total_weight += weight
                if cat_name in attr_1 and cat_name in attr_2 and attr_1[cat_name] == attr_2[cat_name]:
                    weighted_sum += weight
            if total_weight > 0:
                result[n] = weighted_sum / total_weight
            else:
                result[n] = 0.0
        
        return [result[n] for n in n_values]
    
    df = df.with_columns(
        pl.struct([
            f'category_level_{level}_1', 
            f'category_level_{level}_2', 
            'characteristic_attributes_mapping_1', 
            'characteristic_attributes_mapping_2'
        ]).map_elements(code_weighted_tf_idf_characteristics, return_dtype=pl.List(pl.Float64))
        .alias(f'weighted_top_attr_match_list_lvl{level}')
    )
    
    new_cols = []
    new_col_names = []
    for n in n_values:
        col_name = f'weighted_top_attr_match_lvl{level}_n{n}'
        new_cols.append(
            pl.col(f'weighted_top_attr_match_list_lvl{level}').list.get(n_values.index(n)).alias(col_name)
        )
        new_col_names.append(col_name)
    
    df = df.with_columns(new_cols)
    
    df = df.drop(f'weighted_top_attr_match_list_lvl{level}')
    pbar.close()
    
    return df

for level in range(1, 5):
    df = generate_weighted_attribute_matches(df, level)
    gc.collect() 
    
gc.collect()

# lcp&lcs для названий

def longest_common_prefix(str1, str2):
    if str1 is None or str2 is None:
        return None
    
    min_len = min(len(str1), len(str2))
    prefix_len = 0
    
    for i in range(min_len):
        if str1[i] == str2[i]:
            prefix_len += 1
        else:
            break
    
    return prefix_len / min_len if min_len != 0 else 0

def longest_common_subsequence(str1, str2):
    if str1 is None or str2 is None:
        return None
    
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_len = dp[len1][len2]
    return lcs_len / max(len1, len2) if max(len1, len2) != 0 else 0

for col in ('name_norm',):
    total = len(df) * 2
    pbar = tqdm(total=total, desc=f'Calculating LCP and LCS for {col}')
    
    def lcp_with_progress(row):
        pbar.update(1)
        return longest_common_prefix(row[f'{col}_1'], row[f'{col}_2'])
    
    def lcs_with_progress(row):
        pbar.update(1)
        return longest_common_subsequence(row[f'{col}_1'], row[f'{col}_2'])
    
    df = df.with_columns([
        pl.struct([f'{col}_1', f'{col}_2'])
        .map_elements(lcp_with_progress, return_dtype=pl.Float64)
        .alias(f'{col}_lcp'),
        
        pl.struct([f'{col}_1', f'{col}_2'])
        .map_elements(lcs_with_progress, return_dtype=pl.Float64)
        .alias(f'{col}_lcs')
    ])
    pbar.close()
    
gc.collect()

# сходство для списков

def jaccard_similarity(list1, list2):
    if list1 is None or list2 is None:
        return None
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def overlap_coefficient(list1, list2):
    if list1 is None or list2 is None:
        return None
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) != 0 else 0

token_cols = [
    'description_tokens', 
    'description_en_tokens',
    'description_mix_tokens',
    'name_tokens', 
    'name_en_tokens',
    'name_mix_tokens',
    'description_tokens_w_digits', 
    'name_tokens_w_digits'
]

collection_cols = [
    'attr_keys', 
    'attr_vals', 
    'units_name',
    'units_desc',
    'brands_name',
    'brands_desc',
    'colors_name',
    'colors_desc'
]

for col in token_cols:
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} jaccard and overlap scores")

    def jaccard_score(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        try:
            list1 = val1.split()
            list2 = val2.split()
            return jaccard_similarity(list1, list2)
        except (AttributeError, TypeError):
            print('bad!!!')
            return None

    def overlap_score(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        try:
            list1 = val1.split()
            list2 = val2.split()
            return overlap_coefficient(list1, list2)
        except (AttributeError, TypeError):
            print('bad!!!')
            return None

    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(jaccard_score, return_dtype=pl.Float64).alias(f'{col}_jaccard_score'),
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(overlap_score, return_dtype=pl.Float64).alias(f'{col}_overlap_score')
    )
    pbar.close()

for col in collection_cols:
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} jaccard and overlap scores")

    def jaccard_score_collection(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        try:
            return jaccard_similarity(val1, val2)
        except Exception:
            print('bad!!!')
            return None

    def overlap_score_collection(row):
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        try:
            return overlap_coefficient(val1, val2)
        except Exception:
            print('bad!!!')
            return None

    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(jaccard_score_collection, return_dtype=pl.Float64).alias(f'{col}_jaccard_score'),
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(overlap_score_collection, return_dtype=pl.Float64).alias(f'{col}_overlap_score')
    )
    pbar.close()
    
gc.collect()

# количество совпадющих ключей и значений атрибутов

collection_cols_for_common_count = ['attr_keys', 'attr_vals',]

for col in collection_cols_for_common_count:
    total = len(df)
    pbar = tqdm(total=total, desc=f"Processing {col} common elements count")

    def common_elements_count(row):
        pbar.update(1)
        val1 = row[f'{col}_1']
        val2 = row[f'{col}_2']
        if val1 is None or val2 is None:
            return None
        try:
            set1 = set(val1)
            set2 = set(val2)
            return len(set1.intersection(set2))
        except Exception:
            print('bad!!!')
            return None

    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2']).map_elements(common_elements_count, return_dtype=pl.Int64).alias(f'{col}_common_count')
    )
    pbar.close()
    
gc.collect()

# bm25 на name+desc

def tokenize(text):
    if text is None:
        return []
    return re.findall(r'\w+', str(text).lower())

def calculate_bm25_score_name(df):
    total = len(df)
    pbar = tqdm(total=total*2, desc='Calculating BM25Okapi name scores')
    
    def bm25_score_left(row):
        pbar.update(1)
        
        name_1 = row['name_1']
        name_2 = row['name_2']
        
        name_1 = "" if name_1 is None else str(name_1)
        name_2 = "" if name_2 is None else str(name_2)
        
        tokens_1 = tokenize(name_1)
        tokens_2 = tokenize(name_2)
        
        if not tokens_1 or not tokens_2:
            return None
        
        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        
        return float(scores[0]) if len(scores) > 0 else None
    
    def bm25_score_right(row):
        pbar.update(1)
        
        name_1 = row['name_2']
        name_2 = row['name_1']
        
        name_1 = "" if name_1 is None else str(name_1)
        name_2 = "" if name_2 is None else str(name_2)
        
        tokens_1 = tokenize(name_1)
        tokens_2 = tokenize(name_2)
        
        if not tokens_1 or not tokens_2:
            return None
        
        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        
        return float(scores[0]) if len(scores) > 0 else None
    
    df = df.with_columns([
        pl.struct(['name_1', 'name_2'])
        .map_elements(bm25_score_left, return_dtype=pl.Float64)
        .alias('bm25_name_score_left'),
        pl.struct(['name_1', 'name_2'])
        .map_elements(bm25_score_right, return_dtype=pl.Float64)
        .alias('bm25_name_score_right')
    ])
    pbar.close()
    
    return df

def calculate_bm25_score(df):
    total = len(df)
    pbar = tqdm(total=total*2, desc='Calculating BM25Okapi scores')
    
    def bm25_score_left(row):
        pbar.update(1)
        
        name_1 = row['name_1']
        desc_1 = row['description_1']
        name_2 = row['name_2']
        desc_2 = row['description_2']
        
        name_1 = "" if name_1 is None else str(name_1)
        desc_1 = "" if desc_1 is None else str(desc_1)
        name_2 = "" if name_2 is None else str(name_2)
        desc_2 = "" if desc_2 is None else str(desc_2)
        
        combined_1 = f"Name: {name_1}, Desc: {desc_1}"
        combined_2 = f"Name: {name_2}, Desc: {desc_2}"
        
        tokens_1 = tokenize(combined_1)
        tokens_2 = tokenize(combined_2)
        
        if not tokens_1 or not tokens_2:
            return None
        
        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        
        return float(scores[0]) if len(scores) > 0 else None
    
    def bm25_score_right(row):
        pbar.update(1)
        
        name_1 = row['name_2']
        desc_1 = row['description_2']
        name_2 = row['name_1']
        desc_2 = row['description_1']
        
        name_1 = "" if name_1 is None else str(name_1)
        desc_1 = "" if desc_1 is None else str(desc_1)
        name_2 = "" if name_2 is None else str(name_2)
        desc_2 = "" if desc_2 is None else str(desc_2)
        
        combined_1 = f"Name: {name_1}, Desc: {desc_1}"
        combined_2 = f"Name: {name_2}, Desc: {desc_2}"
        
        tokens_1 = tokenize(combined_1)
        tokens_2 = tokenize(combined_2)
        
        if not tokens_1 or not tokens_2:
            return None
        
        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        
        return float(scores[0]) if len(scores) > 0 else None
    
    df = df.with_columns([
        pl.struct(['name_1', 'description_1', 'name_2', 'description_2'])
        .map_elements(bm25_score_left, return_dtype=pl.Float64)
        .alias('bm25_name_desc_score_left'),
        pl.struct(['name_1', 'description_1', 'name_2', 'description_2'])
        .map_elements(bm25_score_right, return_dtype=pl.Float64)
        .alias('bm25_name_desc_score_right')
    ])
    pbar.close()
    
    return df

df = calculate_bm25_score_name(df)
gc.collect()

df = calculate_bm25_score(df)
gc.collect()

# bm25 по key=val атрибутам

def tokenize_keyval(text):
    if text is None:
        return []
    return str(text).strip().split()

def calculate_bm25_score_keyval(df):
    total = len(df)
    pbar = tqdm(total=total*2, desc='Calculating BM25Okapi keyval scores')

    def bm25_score_left(row):
        pbar.update(1)
        keyval_1 = row['concat_keyval_1']
        keyval_2 = row['concat_keyval_2']

        tokens_1 = tokenize_keyval(keyval_1)
        tokens_2 = tokenize_keyval(keyval_2)

        if not tokens_1 or not tokens_2:
            return None

        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        return float(scores[0]) if len(scores) > 0 else None

    def bm25_score_right(row):
        pbar.update(1)
        keyval_1 = row['concat_keyval_2']
        keyval_2 = row['concat_keyval_1']

        tokens_1 = tokenize_keyval(keyval_1)
        tokens_2 = tokenize_keyval(keyval_2)

        if not tokens_1 or not tokens_2:
            return None

        bm25 = BM25Okapi([tokens_2])
        scores = bm25.get_scores(tokens_1)
        return float(scores[0]) if len(scores) > 0 else None

    df = df.with_columns([
        pl.struct(['concat_keyval_1', 'concat_keyval_2'])
        .map_elements(bm25_score_left, return_dtype=pl.Float64)
        .alias('bm25_keyval_score_left'),
        pl.struct(['concat_keyval_1', 'concat_keyval_2'])
        .map_elements(bm25_score_right, return_dtype=pl.Float64)
        .alias('bm25_keyval_score_right')
    ])
    pbar.close()
    return df

df = calculate_bm25_score_keyval(df)
    
gc.collect()

# iou по n-gram

def calculate_ngram_similarities_multiple_cols(df, cols):
    total = len(df) * len(cols)
    pbar = tqdm(total=total, desc='Calculating n-gram Similarities for multiple columns')
    
    def get_ngrams(text, n):
        text = text.lower()
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def ngram_similarities(row, col):
        pbar.update(1)
        val_1 = "" if row[f'{col}_1'] is None else str(row[f'{col}_1'])
        val_2 = "" if row[f'{col}_2'] is None else str(row[f'{col}_2'])
        
        sims = []
        for n in [1, 2, 3, 4, 5, 6, 7]:
            ngrams_1 = set(get_ngrams(val_1, n)) if len(val_1) >= n else set()
            ngrams_2 = set(get_ngrams(val_2, n)) if len(val_2) >= n else set()
            sim = 0.0
            if ngrams_1 and ngrams_2:
                intersection = len(ngrams_1 & ngrams_2)
                union = len(ngrams_1 | ngrams_2)
                sim = intersection / union if union > 0 else 0.0
            sims.append(sim)
        return sims
    
    for col in cols:
        df = df.with_columns(
            pl.struct([f'{col}_1', f'{col}_2'])
            .map_elements(lambda row: ngram_similarities(row, col), return_dtype=pl.List(pl.Float64))
            .alias(f'ngram_similarities_{col}')
        )
    
    for col in cols:
        df = df.with_columns([
            pl.col(f'ngram_similarities_{col}').list.get(0).alias(f'{col}_char_1gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(1).alias(f'{col}_char_2gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(2).alias(f'{col}_char_3gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(3).alias(f'{col}_char_4gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(4).alias(f'{col}_char_5gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(5).alias(f'{col}_char_6gram_iou'),
            pl.col(f'ngram_similarities_{col}').list.get(6).alias(f'{col}_char_7gram_iou'),
        ]).drop(f'ngram_similarities_{col}')
    
    pbar.close()
    return df

columns_ngram = [
    'name',
    'name_en', 
    'name_mix',
    'name_tokens_w_digits',
    'description',
    'description_en', 
    'description_mix', 
    'description_tokens_w_digits'
]

df = calculate_ngram_similarities_multiple_cols(df, columns_ngram)

gc.collect()

# rouge по name+desc, очень долго

def calculate_rouge_metrics(df, col):
    total = len(df)
    pbar = tqdm(total=total, desc='Calculating ROUGE Metrics')
    
    def rouge_scores(row):
        pbar.update(1)
        col_1 = "" if row[f'{col}_1'] is None else str(row[f'{col}_1'])
        col_2 = "" if row[f'{col}_2'] is None else str(row[f'{col}_2'])
        
        if len(col_1) < 1 or len(col_2) < 1:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        try:
            rouge = Rouge()
            scores = rouge.get_scores(col_1, col_2)[0]
            
            rouge1_f = scores['rouge-1']['f']
            rouge1_p = scores['rouge-1']['p']
            rouge1_r = scores['rouge-1']['r']
            
            rouge2_f = scores['rouge-2']['f']
            
            rougeL_f = scores['rouge-l']['f']
            rougeL_r = scores['rouge-l']['r']
            
            return [rouge1_f, rouge1_p, rouge1_r, rouge2_f, rougeL_f, rougeL_r]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    df = df.with_columns(
        pl.struct([f'{col}_1', f'{col}_2'])
        .map_elements(rouge_scores, return_dtype=pl.List(pl.Float64))
        .alias(f'{col}_rouge_metrics')
    )
    
    df = df.with_columns([
        pl.col(f'{col}_rouge_metrics').list.get(0).alias(f'{col}_rouge1_f'),
        pl.col(f'{col}_rouge_metrics').list.get(1).alias(f'{col}_rouge1_precision'),
        pl.col(f'{col}_rouge_metrics').list.get(2).alias(f'{col}_rouge1_recall'),
        pl.col(f'{col}_rouge_metrics').list.get(3).alias(f'{col}_rouge2_f'),
        pl.col(f'{col}_rouge_metrics').list.get(4).alias(f'{col}_rougeL_f'),
        pl.col(f'{col}_rouge_metrics').list.get(5).alias(f'{col}_rougeL_recall')
    ]).drop(f'{col}_rouge_metrics')
    
    pbar.close()
    return df

# for col in ('name', 'description',):
#     df = calculate_rouge_metrics(df, col)
#     gc.collect()

# gc.collect()

train_df = df.filter(df['is_double'] != -1)
test_df = df.filter(df['is_double'] == -1)

# удалим ненужные столбцы

to_drop = [
    'name_1', 'name_2', 
    'description_1', 'description_2', 
    'name_norm_1', 'name_norm_2',
    'description_norm_1', 'description_norm_2', 
    'attr_vals_1', 'attr_vals_2',
    'attr_keys_1', 'attr_keys_2',
    'characteristic_attributes_mapping_1', 'characteristic_attributes_mapping_2',
    'description_tokens_1', 'description_tokens_2',
    'name_tokens_1', 'name_tokens_2', 
    'description_tokens_w_digits_1', 'description_tokens_w_digits_2',
    'price_1', 'price_2',
    'n_images_1', 'n_images_2',
    'name_en_1', 'name_en_2',
    'name_en_norm_1', 'name_en_norm_2',
    'name_en_tokens_1', 'name_en_tokens_2',
    'name_mix_1', 'name_mix_2',
    'name_mix_norm_1', 'name_mix_norm_2',
    'name_mix_tokens_1', 'name_mix_tokens_2',
    'description_en_1', 'description_en_2',
    'description_en_norm_1', 'description_en_norm_2',
    'description_en_tokens_1', 'description_en_tokens_2',
    'description_mix_1', 'description_mix_2',
    'description_mix_norm_1', 'description_mix_norm_2',
    'description_mix_tokens_1', 'description_mix_tokens_2',
    'name_tokens_len_1', 'name_tokens_len_2',
    'description_tokens_len_1', 'description_tokens_len_2',
    'name_en_tokens_len_1', 'name_en_tokens_len_2',
    'description_en_tokens_len_1', 'description_en_tokens_len_2',
    'name_mix_tokens_len_1', 'name_mix_tokens_len_2',
    'description_mix_tokens_len_1', 'description_mix_tokens_len_2',
    'attr_keys_len_1', 'attr_keys_len_2',
    'attr_vals_len_1', 'attr_vals_len_2',
    'units_name_1', 'units_name_2',
    'units_desc_1', 'units_desc_2',
    'brands_name_1', 'brands_name_2',
    'brands_desc_1', 'brands_desc_2',
    'colors_name_1', 'colors_name_2',
    'colors_desc_1', 'colors_desc_2',
    'name_tokens_w_digits_1', 'name_tokens_w_digits_2',
    'concat_keyval_1', 'concat_keyval_2'
]

train_df = train_df.drop(to_drop)
test_df = test_df.drop(to_drop)

train_df.write_parquet('../data/merged-with-features/train_df.parquet')
test_df.write_parquet('../data/merged-with-features/test_df.parquet')