import polars as pl
import re
import numpy as np
import json
import pickle
import gc
import re

from collections import Counter
from typing import List, Dict, Callable
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

import scipy.sparse as sp
from tqdm.notebook import tqdm

pl.enable_string_cache()

MAIN_DIR = Path('../data/avito-orig-data/')
FIX_ND_DIR = Path('../data/fix-letters-kenlm/')

FILE_PATHES = [
    MAIN_DIR / Path('train_part_0001.snappy.parquet'),
    MAIN_DIR / Path('train_part_0002.snappy.parquet'),
    MAIN_DIR / Path('train_part_0003.snappy.parquet'),
    MAIN_DIR / Path('train_part_0004.snappy.parquet'),
    MAIN_DIR / Path('test_part_0001.snappy.parquet'),
    MAIN_DIR / Path('test_part_0002.snappy.parquet'),
]

BnC_DIR = Path('../data/brands-and-colors/')

RENAME_MAPPING = {
    'base_item_id': 'variantid_1', 
    'cand_item_id': 'variantid_2',
    'base_title': 'name_1',
    'cand_title': 'name_2',
    'base_description': 'description_1',
    'cand_description': 'description_2',
    'base_category_name': 'category_level_1_1',
    'cand_category_name': 'category_level_1_2',
    'base_subcategory_name': 'category_level_2_1',
    'cand_subcategory_name': 'category_level_2_2',
    'base_param1': 'category_level_3_1',
    'cand_param1': 'category_level_3_2',
    'base_param2': 'category_level_4_1',
    'cand_param2': 'category_level_4_2',
    'base_json_params': 'characteristic_attributes_mapping_1',
    'cand_json_params': 'characteristic_attributes_mapping_2',
    'base_count_images': 'n_images_1',
    'cand_count_images': 'n_images_2',
    'base_price': 'price_1',
    'cand_price': 'price_2'
}

IDS = ['variantid_1', 'variantid_2']
FOR_SPLIT = ['group_id', 'action_date']
IMAGE_PATHS = ['base_title_image', 'cand_title_image']
BINARY_FEATURES = ['is_same_location', 'is_same_region']
TARGET = 'is_double'

full_df = pl.DataFrame()

for file in tqdm(FILE_PATHES):
    chunk = pl.read_parquet(file)
    if 'is_double' not in chunk.columns:
        chunk = chunk.with_columns(
            is_double=pl.lit(-1),
            action_date=pl.lit(-1),
            group_id=pl.lit(-1)
        )
        chunk = chunk.select(full_df.columns)
    full_df = pl.concat([full_df, chunk], how='diagonal_relaxed')
    print(f'{full_df.shape=}')

del chunk 
gc.collect()
full_df = full_df.rename(mapping=RENAME_MAPPING)

with open(BnC_DIR / Path('brands.pkl'), 'rb') as f: # потом с ллм отфильтровать бренды
    brands = pickle.load(f)

colors = pl.read_csv(BnC_DIR / Path('ruscorpora_content_colors.csv'), separator=';')['word_0'].to_list()

data_1 = full_df.select(
    [col for col in full_df.columns if col.endswith('_1')]
).rename({col: col[:-2] for col in full_df.columns if col.endswith('_1')}).unique()

data_2 = full_df.select(
    [col for col in full_df.columns if col.endswith('_2')]
).rename({col: col[:-2] for col in full_df.columns if col.endswith('_2')}).unique()

data = pl.concat([data_1, data_2], how='vertical').unique()

del data_1, data_2
gc.collect()

print(f'{data.shape=}')

names_and_descs_kenlm = pl.read_parquet(FIX_ND_DIR / Path('avito_bad_people_fixed.parquet'))
print(f'{names_and_descs_kenlm.shape=}')

names_and_descs_kenlm = names_and_descs_kenlm.rename(mapping={
    'name_1_fixed': 'name_1',
    'desc_1_fixed': 'description_1',
    'name_1_ratio_fixed': 'name_broken_perc_1',
    'desc_1_ratio_fixed': 'description_broken_perc_1',
    'name_2_fixed': 'name_2',
    'desc_2_fixed': 'description_2',
    'name_2_ratio_fixed': 'name_broken_perc_2',
    'desc_2_ratio_fixed': 'description_broken_perc_2',
})

full_df = full_df.sort(by=['variantid_1', 'variantid_2'])
names_and_descs_kenlm = names_and_descs_kenlm.sort(by=['variantid_1', 'variantid_2'])

full_df = full_df.with_columns(names_and_descs_kenlm["name_1"].alias("name_1"))
full_df = full_df.with_columns(names_and_descs_kenlm["description_1"].alias("description_1"))
full_df = full_df.with_columns(names_and_descs_kenlm["name_broken_perc_1"].alias("name_broken_perc_1"))
full_df = full_df.with_columns(names_and_descs_kenlm["description_broken_perc_1"].alias("description_broken_perc_1"))

full_df = full_df.with_columns(names_and_descs_kenlm["name_2"].alias("name_2"))
full_df = full_df.with_columns(names_and_descs_kenlm["description_2"].alias("description_2"))
full_df = full_df.with_columns(names_and_descs_kenlm["name_broken_perc_2"].alias("name_broken_perc_2"))
full_df = full_df.with_columns(names_and_descs_kenlm["description_broken_perc_2"].alias("description_broken_perc_2"))

full_df.shape, names_and_descs_kenlm.shape

def remove_html_tags_and_emoji(text: str) -> str:
    if text is None:
        return None
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('\n', ' ')
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def filter_en_text(text: str) -> str:
    pattern = r"\b[a-zA-Z]+\b"
    res = ' '.join(re.findall(pattern, text.lower()))
    return res

def extract_mixed_words(s: str) -> str:
    pattern = r'\b(?=[A-Za-zА-Яа-яЁё]*\d)(?=\d*[A-Za-zА-Яа-яЁё])[A-Za-zА-Яа-яЁё\d]+\b'
    matches = re.findall(pattern, s)
    return ' '.join(matches)

def mixed_language_words_percentage(text: str) -> float:
    words = text.split()
    
    def is_mixed_language(word: str) -> bool:
        if re.search(r'\d', word):
            return False
        has_russian = bool(re.search(r'[а-яА-Я]', word))
        has_english = bool(re.search(r'[a-zA-Z]', word))
        return has_russian and has_english
    
    mixed_words_count = sum(is_mixed_language(word) for word in words)
    
    if len(words) == 0:
        return 0.0
    return (mixed_words_count / len(words)) * 100

def normalize(text: str) -> str:
    if text is None:
        return None
    text = text.lower()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append(' ')
    tokens = ''.join(chars).split() 
    return '_'.join(tokens)

# переписать на kenlm [DONE, юзаются сразу пофикшенные названия и описания]
# def fix_key_layout(text: str, thold: float = 0.6) -> str:
#     en_ru_mapping = {
#         'e': 'е',
#         'y': 'у',
#         'o': 'о',
#         'p': 'р',
#         'a': 'а',
#         'k': 'к',
#         'x': 'х',
#         'c': 'с',
#         'E': 'Е',
#         'T': 'Т',
#         'O': 'О',
#         'P': 'Р',
#         'A': 'А',
#         'H': 'Н',
#         'K': 'К',
#         'X': 'Х',
#         'C': 'С',
#         'B': 'В',
#         'M': 'М',
#     }

#     def is_english_letter(char: str) -> bool:
#         return 'a' <= char <= 'z' or 'A' <= char <= 'Z'

#     def should_apply_mapping(word: str, thold: float = 0.6) -> float:
#         if not word:
#             return False
#         total_letters = sum(c.isalpha() for c in word)
#         if total_letters == 0:
#             return False
#         english_letters = sum(is_english_letter(c) for c in word)
#         return (english_letters / total_letters) < thold

#     words = text.split()
#     fixed_words = []
#     for word in words:
#         if should_apply_mapping(word):
#             fixed_word = ''.join(en_ru_mapping.get(char, char) for char in word)
#         else:
#             fixed_word = word
#         fixed_words.append(fixed_word)

#     res = ' '.join(fixed_words)
#     res = res.replace('нa', 'на').replace(' c ', ' с ').replace(
#         ' cо ', ' со ').replace(' сo ', ' со ').replace(
#         ' co ', ' со ').replace(' вo ', ' во ').replace(
#         ' кo ', ' ко ').replace(' o ', ' о ').replace(
#         ' oб ', ' об ').replace(' oт ', ' от ').replace(
#         ' зa ', ' за ').replace(' пo ', ' по ').replace(
#         ' дo ', ' до ').replace(' y ', ' у ').replace(
#         ' cм ', ' см ').replace(' гp', ' гр ').replace(
#         ' пpo ', ' про ').replace(' дa ', ' да ').replace(
#         ' нe ', ' не ').replace(' тo ', ' то ').replace(
#         ' жe ', ' же ').replace(' pyб ', ' руб ').replace(
#         ' eд ', ' ед ').replace(' oна ', ' она ').replace(
#         ' онa ', ' она ').replace(' oнa ', ' она ').replace(
#         ' oн ', ' он ').replace(' eго ', ' его ').replace(
#         ' егo ', ' его ').replace(' eгo ', ' его ').replace(
#         ' ниx ', ' них ').replace(' иx ', ' их ').replace(
#         ' вcе ', ' все ').replace(' всe ', ' все ').replace(' вce ', ' все ')
#     return res

def normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('name_1').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('name_1')
    )
    # df = df.with_columns(
    #     pl.col('name_1').map_elements(fix_key_layout, return_dtype=pl.Utf8).alias('name_1')
    # )
    df = df.with_columns(
        pl.col('name_1').map_elements(normalize, return_dtype=pl.Utf8).alias('name_norm_1'),
        pl.col('name_1').str.strip_chars().str.to_lowercase().alias('name_tokens_1'),
    )
    df = df.with_columns(
        pl.col('name_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_1'),
    )
    
    df = df.with_columns(
        pl.col('name_2').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('name_2')
    )
    # df = df.with_columns(
    #     pl.col('name_2').map_elements(fix_key_layout, return_dtype=pl.Utf8).alias('name_2')
    # )
    df = df.with_columns(
        pl.col('name_2').map_elements(normalize, return_dtype=pl.Utf8).alias('name_norm_2'),
        pl.col('name_2').str.strip_chars().str.to_lowercase().alias('name_tokens_2'),
    )
    df = df.with_columns(
        pl.col('name_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_2'),
    )
    
    return df


def normalize_en_names(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('name_1').map_elements(filter_en_text, return_dtype=pl.Utf8).alias('name_en_1')
    )
    df = df.with_columns(
        pl.col('name_en_1').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('name_en_1')
    )
    df = df.with_columns(
        pl.col('name_en_1').map_elements(normalize, return_dtype=pl.Utf8).alias('name_en_norm_1'),
        pl.col('name_en_1').str.strip_chars().str.to_lowercase().alias('name_en_tokens_1'),
    )
    df = df.with_columns(
        pl.col('name_en_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_en_1'),
    )
    
    df = df.with_columns(
        pl.col('name_2').map_elements(filter_en_text, return_dtype=pl.Utf8).alias('name_en_2')
    )
    df = df.with_columns(
        pl.col('name_en_2').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('name_en_2')
    )
    df = df.with_columns(
        pl.col('name_en_2').map_elements(normalize, return_dtype=pl.Utf8).alias('name_en_norm_2'),
        pl.col('name_en_2').str.strip_chars().str.to_lowercase().alias('name_en_tokens_2'),
    )
    df = df.with_columns(
        pl.col('name_en_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_en_2'),
    )
    
    return df

def normalize_mixed_names(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('name_1').map_elements(extract_mixed_words, return_dtype=pl.Utf8).alias('name_mix_1')
    )
    df = df.with_columns(
        pl.col('name_mix_1').map_elements(normalize, return_dtype=pl.Utf8).alias('name_mix_norm_1'),
        pl.col('name_mix_1').str.strip_chars().str.to_lowercase().alias('name_mix_tokens_1'),
    )
    df = df.with_columns(
        pl.col('name_mix_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_mix_1'),
    )

    df = df.with_columns(
        pl.col('name_2').map_elements(extract_mixed_words, return_dtype=pl.Utf8).alias('name_mix_2')
    )
    df = df.with_columns(
        pl.col('name_mix_2').map_elements(normalize, return_dtype=pl.Utf8).alias('name_mix_norm_2'),
        pl.col('name_mix_2').str.strip_chars().str.to_lowercase().alias('name_mix_tokens_2'),
    )
    df = df.with_columns(
        pl.col('name_mix_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('name_mix_2'),
    )
    
    return df

def normalize_desc(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('description_1').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('description_1')
    )
    # df = df.with_columns(
    #     pl.col('description_1').map_elements(fix_key_layout, return_dtype=pl.Utf8).alias('description_1')
    # )
    df = df.with_columns(
        pl.col('description_1').map_elements(normalize, return_dtype=pl.Utf8).alias('description_norm_1'),
        pl.col('description_1').str.strip_chars().str.to_lowercase().alias('description_tokens_1'),
    )
    df = df.with_columns(
        pl.col('description_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_1'),
    )
    
    df = df.with_columns(
        pl.col('description_2').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('description_2')
    )
    # df = df.with_columns(
    #     pl.col('description_2').map_elements(fix_key_layout, return_dtype=pl.Utf8).alias('description_2')
    # )
    df = df.with_columns(
        pl.col('description_2').map_elements(normalize, return_dtype=pl.Utf8).alias('description_norm_2'),
        pl.col('description_2').str.strip_chars().str.to_lowercase().alias('description_tokens_2'),
    )
    df = df.with_columns(
        pl.col('description_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_2'),
    )
    
    return df

def normalize_en_desc(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('description_1').map_elements(filter_en_text, return_dtype=pl.Utf8).alias('description_en_1')
    )
    df = df.with_columns(
        pl.col('description_en_1').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('description_en_1')
    )
    df = df.with_columns(
        pl.col('description_en_1').map_elements(normalize, return_dtype=pl.Utf8).alias('description_en_norm_1'),
        pl.col('description_en_1').str.strip_chars().str.to_lowercase().alias('description_en_tokens_1'),
    )
    df = df.with_columns(
        pl.col('description_en_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_en_1'),
    )

    df = df.with_columns(
        pl.col('description_2').map_elements(filter_en_text, return_dtype=pl.Utf8).alias('description_en_2')
    )
    df = df.with_columns(
        pl.col('description_en_2').map_elements(remove_html_tags_and_emoji, return_dtype=pl.Utf8).alias('description_en_2')
    )
    df = df.with_columns(
        pl.col('description_en_2').map_elements(normalize, return_dtype=pl.Utf8).alias('description_en_norm_2'),
        pl.col('description_en_2').str.strip_chars().str.to_lowercase().alias('description_en_tokens_2'),
    )
    df = df.with_columns(
        pl.col('description_en_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_en_2'),
    )
    
    return df

def normalize_mixed_desc(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('description_1').map_elements(extract_mixed_words, return_dtype=pl.Utf8).alias('description_mix_1')
    )
    df = df.with_columns(
        pl.col('description_mix_1').map_elements(normalize, return_dtype=pl.Utf8).alias('description_mix_norm_1'),
        pl.col('description_mix_1').str.strip_chars().str.to_lowercase().alias('description_mix_tokens_1'),
    )
    df = df.with_columns(
        pl.col('description_mix_tokens_1').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_mix_1'),
    )

    df = df.with_columns(
        pl.col('description_2').map_elements(extract_mixed_words, return_dtype=pl.Utf8).alias('description_mix_2')
    )
    df = df.with_columns(
        pl.col('description_mix_2').map_elements(normalize, return_dtype=pl.Utf8).alias('description_mix_norm_2'),
        pl.col('description_mix_2').str.strip_chars().str.to_lowercase().alias('description_mix_tokens_2'),
    )
    df = df.with_columns(
        pl.col('description_mix_tokens_2').map_elements(lambda x: ' '.join(x.split()), return_dtype=pl.Utf8).alias('description_mix_2'),
    )
    
    return df

def normalize_characteristic_attributes(df: pl.DataFrame) -> pl.DataFrame:
    def flatten_json(d: dict, parent_key: str = ''):
        items = {}
        for k, v in d.items():
            if isinstance(v, list):
                for i in v:
                    if isinstance(i, dict):
                        items.update(flatten_json(i))
                    else:
                        items[k] = ','.join(str(x) for x in v)
            elif isinstance(v, dict):
                items.update(flatten_json(v))
            else:
                items[k] = str(v)
        return items

    def process(x):
        flat = flatten_json(json.loads(x))
        flat_json = json.dumps(flat)
        concat_keyval = ' '.join([f"{k}={v}" for k, v in flat.items()])
        return flat_json, concat_keyval

    df = df.with_columns([
        pl.col('characteristic_attributes_mapping_1')
        .map_elements(lambda x: process(x)[0], return_dtype=pl.String)
        .alias('characteristic_attributes_mapping_1'),
        pl.col('characteristic_attributes_mapping_1')
        .map_elements(lambda x: process(x)[1], return_dtype=pl.String)
        .alias('concat_keyval_1')
    ])

    df = df.with_columns([
        pl.col('characteristic_attributes_mapping_2')
        .map_elements(lambda x: process(x)[0], return_dtype=pl.String)
        .alias('characteristic_attributes_mapping_2'),
        pl.col('characteristic_attributes_mapping_2')
        .map_elements(lambda x: process(x)[1], return_dtype=pl.String)
        .alias('concat_keyval_2')
    ])
    
    return df

def normalize_characteristic_attributes_prikol(df: pl.DataFrame) -> pl.DataFrame:
    def flatten_json(d: dict, parent_key: str = ''):
        items = {}
        for k, v in d.items():
            if isinstance(v, list):
                for i in v:
                    if isinstance(i, dict):
                        items.update(flatten_json(i))
                    else:
                        items[k] = ','.join(str(x) for x in v)
            elif isinstance(v, dict):
                items.update(flatten_json(v))
            else:
                items[k] = str(v)
        return items

    def process(x):
        flat = flatten_json(json.loads(x))
        flat_json = json.dumps(flat)
        concat_keyval = ' '.join([f"{k}={v}" for k, v in flat.items()])
        return flat_json, concat_keyval

    return df.with_columns([
        pl.col('characteristic_attributes_mapping')
        .map_elements(lambda x: process(x)[0], return_dtype=pl.String)
        .alias('characteristic_attributes_mapping')
    ])

def get_kv_from_attrs(df: pl.DataFrame) -> pl.DataFrame:
    def get_list_of_values(values: list) -> list:
        values_to_return = []
        for value in values:
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    dict_values = [str(i) if i is not None else 'none' for i in value[0].values()]
                    values_to_return.extend(dict_values)
                else:
                    values_to_return.extend([str(i) for i in value])
            else:
                values_to_return.append(str(value))
        return values_to_return
    
    def get_list_of_keys(json_dict: dict) -> list:
        values_to_return = []
        for key, value in json_dict.items():
            values_to_return.append(str(key))
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    dict_keys = [str(i) for i in value[0].keys()]
                    values_to_return.extend(dict_keys)
        return values_to_return

    df = df.with_columns(
        pl.col('characteristic_attributes_mapping_1')
            .map_elements(
                lambda x: get_list_of_keys(json.loads(x)),
                return_dtype=pl.List(pl.Utf8)
        ).alias('attr_keys_1'),
        pl.col('characteristic_attributes_mapping_1')
            .map_elements(
                lambda x: get_list_of_values(list(json.loads(x).values())),
                return_dtype=pl.List(pl.Utf8)
        ).alias('attr_vals_1')
    )
    
    df = df.with_columns(
        pl.col('characteristic_attributes_mapping_2')
            .map_elements(
                lambda x: get_list_of_keys(json.loads(x)),
                return_dtype=pl.List(pl.Utf8)
        ).alias('attr_keys_2'),
        pl.col('characteristic_attributes_mapping_2')
            .map_elements(
                lambda x: get_list_of_values(list(json.loads(x).values())),
                return_dtype=pl.List(pl.Utf8)
        ).alias('attr_vals_2')
    )
    
    return df

def get_lengths(df: pl.DataFrame) -> pl.DataFrame: 
    df = df.with_columns(
        (pl.col('name_tokens_1').str.count_matches(' ') + 1).alias('name_tokens_len_1'),
        (pl.col('description_tokens_1').str.count_matches(' ') + 1).alias('description_tokens_len_1'),
        (pl.col('name_en_tokens_1').str.count_matches(' ') + 1).alias('name_en_tokens_len_1'),
        (pl.col('description_en_tokens_1').str.count_matches(' ') + 1).alias('description_en_tokens_len_1'),
        (pl.col('name_mix_tokens_1').str.count_matches(' ') + 1).alias('name_mix_tokens_len_1'),
        (pl.col('description_mix_tokens_1').str.count_matches(' ') + 1).alias('description_mix_tokens_len_1'),
        pl.col('attr_keys_1').list.len().alias('attr_keys_len_1'),
        pl.col('attr_vals_1').list.len().alias('attr_vals_len_1'),
    )
    
    df = df.with_columns(
        (pl.col('name_tokens_2').str.count_matches(' ') + 1).alias('name_tokens_len_2'),
        (pl.col('description_tokens_2').str.count_matches(' ') + 1).alias('description_tokens_len_2'),
        (pl.col('name_en_tokens_2').str.count_matches(' ') + 1).alias('name_en_tokens_len_2'),
        (pl.col('description_en_tokens_2').str.count_matches(' ') + 1).alias('description_en_tokens_len_2'),
        (pl.col('name_mix_tokens_2').str.count_matches(' ') + 1).alias('name_mix_tokens_len_2'),
        (pl.col('description_mix_tokens_2').str.count_matches(' ') + 1).alias('description_mix_tokens_len_2'),
        pl.col('attr_keys_2').list.len().alias('attr_keys_len_2'),
        pl.col('attr_vals_2').list.len().alias('attr_vals_len_2'),
    )
    
    return df

def get_digits_elements(df: pl.DataFrame) -> pl.DataFrame:
    def extract_tokens_with_digits(text: str) -> str:
        if not isinstance(text, str):
            return ''
        return ' '.join([token for token in text.split() if len(re.findall(r'\d', token)) > 2])
    
    df = df.with_columns([
        pl.col('name_tokens_1').map_elements(extract_tokens_with_digits, return_dtype=pl.Utf8).alias('name_tokens_w_digits_1'),
        pl.col('description_tokens_1').map_elements(extract_tokens_with_digits, return_dtype=pl.Utf8).alias('description_tokens_w_digits_1'),
    ])

    df = df.with_columns([
        pl.col('name_tokens_2').map_elements(extract_tokens_with_digits, return_dtype=pl.Utf8).alias('name_tokens_w_digits_2'),
        pl.col('description_tokens_2').map_elements(extract_tokens_with_digits, return_dtype=pl.Utf8).alias('description_tokens_w_digits_2'),
    ])
    
    return df

def extract_brands(df: pl.DataFrame) -> pl.DataFrame:
    global brands
    brands_set = set(brands)

    def find_brands(s: str) -> list[str]:
        if s is None:
            return []
        words = s.split()
        return list(set([word for word in words if word in brands_set]))

    df = df.with_columns(
        pl.col('name_1').map_elements(find_brands, return_dtype=pl.List(pl.Utf8)).alias('brands_name_1')
    )
    df = df.with_columns(
        pl.col('description_1').map_elements(find_brands, return_dtype=pl.List(pl.Utf8)).alias('brands_desc_1')
    )

    df = df.with_columns(
        pl.col('name_2').map_elements(find_brands, return_dtype=pl.List(pl.Utf8)).alias('brands_name_2')
    )
    df = df.with_columns(
        pl.col('description_2').map_elements(find_brands, return_dtype=pl.List(pl.Utf8)).alias('brands_desc_2')
    )
    
    return df

def extract_colors(df: pl.DataFrame) -> pl.DataFrame:
    global colors
    colors_set = set(colors)

    def find_colors(s: str) -> list[str]:
        if s is None:
            return []
        words = s.split()
        return list(set([word for word in words if word in colors_set]))

    df = df.with_columns(
        pl.col('name_1').map_elements(find_colors, return_dtype=pl.List(pl.Utf8)).alias('colors_name_1')
    )
    df = df.with_columns(
        pl.col('description_1').map_elements(find_colors, return_dtype=pl.List(pl.Utf8)).alias('colors_desc_1')
    )
    
    df = df.with_columns(
        pl.col('name_2').map_elements(find_colors, return_dtype=pl.List(pl.Utf8)).alias('colors_name_2')
    )
    df = df.with_columns(
        pl.col('description_2').map_elements(find_colors, return_dtype=pl.List(pl.Utf8)).alias('colors_desc_2')
    )
    
    return df

def find_units(s: str) -> list[str]:
    if s is None:
        return []
        
    units = [
        'мм', 'м', 'см', 'дм', 'км', 'нм', 'мкм', 'дюйм', 'сантиметр', 'миллиметр', 'километр', 'нанометр', 'km',
        'мм²', 'см²', 'дм²', 'м²', 'км²',
        'м³', 'см³', 'мм³',
        'г', 'кг', 'мг', 'т', 'kg', 'g', 'грамм', 'килограмм', 'миллиграмм', 'mg',
        'л', 'мл', 'куб\.см', 'куб\.м', 'литр', 'миллилитр', 'ml',
        'гб', 'гбит', 'мб', 'мбит', 'кб', 'кбит', 'тб', 'тбит', 'байт', 'бит', 'гигабайт', 'гигабит', 'мегабайт', 'gb', 'kb', 'mb',
        'час', 'ч', 'мин', 'сек', 'с', 'минут', 'секунд', 'min', 'sec', 'h',
        'в', 'квт', 'мвт', 'вт', 'ва', 'ква', 'мва', 'а', 'ма', 'ка', 'мка', 'вольт', 'ампер', 'ампер-час', 'ач', 'мач', 'ватт', 'mah', 'w',
        'ом', 'ohm', 'Ω', 'mΩ',
        'ф', 'мкф', 'нф', 'пф', 
        'гц', 'кгц', 'мгц', 'ггц', 'герц', 'килогерц', 'мегагерц', 'гигагерц', 'hz', 'khz', 'mhz', 'ghz',
        'дб', 'децибел', 'db',
        'бар', 'паскаль', 'па', 'гпа', 'кпа', 'атм',
        'градус', '°', 'рад', 'радиан',
        '°c', '°f', 'град', 'цельсий', 'фаренгейт',
        'процент', '%',
        'шт', 'ед', 'штук',
        'дж', 'кдж', 'ккал',
        'моль',
        'н', 'кн',
        'лс', 'об/мин', 'км/ч', 'м/с', 'лошадиных сил', 'm/h', 'km/h', 'kmh', 'км/с', 'мм/с', 'миль/ч', 
        'г/см3', 'кг/м3',
        'люмен', 'дптр',
        'пиксель', 'пикс', 'px', 'dpi', 'ppi', 'кадр/с', 'fps',
        'flops', 'gflops', 'tflops', 'mips', 'ipc',
        '$', '€', '£', '¥', 'руб.', 'р.', 'rub', 'usd', 'eur',
        'ядро', 'ядер', 'поток', 'потоков', 'thread', 'core', 'операций/с', 'op/s',
        'дюймов', 'inch',
        'MP', 'Мп', 'mpx', 'мегапиксель', 'мегапикселей',
    ]
    
    units_sorted = sorted(units, key=len, reverse=True)
    safe_units = [re.escape(u) for u in units_sorted]
    units_pattern = '|'.join(safe_units)
    
    num_pattern = r'\d+(?:[\.,]\d+)?'
    pattern_string = rf'({num_pattern})(\s)?({units_pattern})\b'
    
    regex = re.compile(pattern_string, re.IGNORECASE)
    matches = regex.findall(s)

    unit_res = [f'{x[0]} {x[2]}' for x in matches]

    size_pattern = r'(\d+)\s*[xх*]\s*(\d+)(?:\s*[xх*]\s*(\d+))?'
    matches = re.findall(size_pattern, s)
    dim_res = []
    for match in matches:
        if match[2]: # 3d
            dim_res.append(f'{match[0]}x{match[1]}x{match[2]}') 
        else: # 2d
            dim_res.append(f'{match[0]}x{match[1]}')
    
    return list(set(unit_res + dim_res))

def extract_units(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col('name_1').map_elements(find_units, return_dtype=pl.List(pl.Utf8)).alias('units_name_1')
    )
    df = df.with_columns(
        pl.col('description_1').map_elements(find_units, return_dtype=pl.List(pl.Utf8)).alias('units_desc_1')
    )
    
    df = df.with_columns(
        pl.col('name_2').map_elements(find_units, return_dtype=pl.List(pl.Utf8)).alias('units_name_2')
    )
    df = df.with_columns(
        pl.col('description_2').map_elements(find_units, return_dtype=pl.List(pl.Utf8)).alias('units_desc_2')
    )
    
    return df

def fit_tfidf_vectorizers_combined(data: pl.DataFrame) -> Dict[str, TfidfVectorizer]:
    global groups, tfidf_vectorizers
    
    for group_name, group_info in groups.items():
        columns = group_info['columns']
        params = group_info['params']
        
        combined_texts = []
        for col in columns:
            curr_data = [str(x) for x in data[col].to_list() if x is not None]
            combined_texts.extend(curr_data)
        
        unique_texts = list(set(combined_texts))
        vectorizer = TfidfVectorizer(**params)
        vectorizer.fit(unique_texts)
        tfidf_vectorizers[group_name] = vectorizer
        
        with open(f'{group_name}_tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    
    return data

def tfidf_emb_gen_combined(
    data: pl.DataFrame,
    batch_size: int = 5000
) -> pl.DataFrame:
    global groups, tfidf_vectorizers
    
    for group_name, group_info in groups.items():
        columns = group_info['columns']
        vectorizer = tfidf_vectorizers[group_name]
        
        for col in columns:
            tfidf_col_sparse = []
            total_rows = len(data)
            
            for start in tqdm(
                range(0, total_rows, batch_size),
                desc=f'Processing {col}',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                colour='yellow'
            ):
                end = min(start + batch_size, total_rows)
                
                batch_texts = (data.slice(start, end - start)
                    .get_column(col)
                    .cast(pl.Utf8)
                    .to_list()
                )
                tfidf_batch_sparse = vectorizer.transform(batch_texts)
                tfidf_col_sparse.append(tfidf_batch_sparse)
            
            tfidf_col_sparse = sp.vstack(tfidf_col_sparse)
            sparse_rows = [row for row in tfidf_col_sparse]
            
            data = data.with_columns(pl.Series(f'{col}_tfidf', sparse_rows))
    
    return data

def find_anti_words(row):
    words1 = set(re.findall(r'([a-z]+)', row['name_1'].lower()))
    words2 = set(re.findall(r'([a-z]+)', row['name_2'].lower()))
    xor_words = words1.symmetric_difference(words2)
    return list(xor_words)

def calculate_idf_by_category_level(data, level):
    kv_idf_by_models = {}
    category_col = f'category_level_{level}'
    unique_categories = data.select(category_col).unique().to_series().to_list()

    for group_cat in unique_categories:
        sub_group_df = data.filter(pl.col(category_col) == group_cat)
        series_charactes = sub_group_df.select('characteristic_attributes_mapping').to_series()
        
        keys_idf = Counter()
        values_idf = {}
        
        for row in series_charactes:
            if row is None:
                continue
            row = json.loads(row)
            for k, v in row.items():
                keys_idf[k] += 1
                if k not in values_idf:
                    values_idf[k] = Counter()
                values_idf[k][v] += 1
                        
        kv_idf_by_models[group_cat] = (keys_idf, values_idf, len(series_charactes))

    pop_characts_tf_idf = {}

    for group_cat in unique_categories:
        characters = kv_idf_by_models[group_cat][0]
        items_in_group = kv_idf_by_models[group_cat][2]        
        characters_list = []
        
        for k in characters:
            counts_k = kv_idf_by_models[group_cat][0][k]
            key_tf = counts_k / items_in_group
            counts_v = len(kv_idf_by_models[group_cat][1][k].keys())
            if counts_v == 0:
                print(f'{k=} in {group_cat=} has 0 unique values')
                continue
            value_idf = np.log(counts_k / counts_v)
            tf_idf = key_tf * value_idf
            characters_list.append((tf_idf, k))
            
        pop_characts_tf_idf[group_cat] = sorted(characters_list, reverse=True)

    with open(f'../data/preprocessed/pop_characts_tf_idf_level_{level}.pkl', 'wb') as file:
        pickle.dump(pop_characts_tf_idf, file)

    return pop_characts_tf_idf

# full_df = full_df.sample(n=1000, shuffle=True)

processing_pipeline = [
    normalize_names,
    normalize_en_names,
    normalize_mixed_names,
    normalize_desc,
    normalize_en_desc,
    normalize_mixed_desc,
    get_kv_from_attrs,
    normalize_characteristic_attributes,
    get_digits_elements,
    get_lengths,
    extract_units,
    extract_brands,
    extract_colors,
    fit_tfidf_vectorizers_combined,
    tfidf_emb_gen_combined
]

def preprocessing(data: pl.DataFrame, pipeline: List[Callable]) -> pl.DataFrame:
    with tqdm(pipeline, desc='Data Preprocessing') as pbar:
        for func in pbar:
            pbar.set_postfix({'current_operation': func.__name__})
            data = func(data)
    return data

tfidf_vectorizers = {}    

groups = {
    'concat_keyval': {
        'columns': ['concat_keyval_1', 'concat_keyval_2'],
        'params': {'tokenizer': str.split, 'preprocessor': None, 'token_pattern': None}
    },
    'name': {
        'columns': ['name_1', 'name_2'],
        'params': {}
    },
    'description': {
        'columns': ['description_1', 'description_2'],
        'params': {}
    }
}

preprocessed_df = preprocessing(full_df, processing_pipeline)

preprocessed_df.drop(
    [c for c in preprocessed_df.columns if 'tfidf' in c]
).write_parquet('../data/preprocessed/all_products_preprocessed.parquet')

for col in [c for c in preprocessed_df.columns if 'tfidf' in c]:
    matrices = preprocessed_df[col].to_list()
    sp.save_npz(f'../data/preprocessed/{col}_matrices.npz', sp.vstack(matrices))

filtered = full_df.filter(pl.col('is_double') == 0)
result = (
    filtered
    .with_columns(
        pl.struct(['name_1', 'name_2']).map_elements(find_anti_words, return_dtype=pl.List(pl.Utf8)).alias('xor_words')
    )
)

anti_words = Counter()

for words in result['xor_words']:
    anti_words.update(words)

filtered_anti_words = Counter({word: count for word, count in anti_words.items() if len(word) > 2})

with open('../data/preprocessed/anti_words.pkl', 'wb') as file:
    pickle.dump(anti_words, file)

with open('../data/preprocessed/filtered_anti_words.pkl', 'wb') as file:
    pickle.dump(filtered_anti_words, file)

filtered_anti_words.most_common(10)

data = normalize_characteristic_attributes_prikol(data)

pop_characts_tf_idf_c1 = calculate_idf_by_category_level(data, 1)
pop_characts_tf_idf_c2 = calculate_idf_by_category_level(data, 2)
pop_characts_tf_idf_c3 = calculate_idf_by_category_level(data, 3)
pop_characts_tf_idf_c4 = calculate_idf_by_category_level(data, 4)