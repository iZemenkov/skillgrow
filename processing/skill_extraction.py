# processing/skill_extraction.py
"""
Извлечение и векторизация навыков для вакансий.
- Источник: CSV (локальный путь или URL), по умолчанию ../data/raw_data/vacancies_master.csv
- Результаты:
  * ../data/processed/extracted_skills.csv       (id, description, skills_list)
  * ../data/processed/embeddings.parquet         (id, desc_emb, skills_emb)
Запуск:
  python processing/skill_extraction.py --src ../data/raw_data/vacancies_master.csv --max_rows 0
"""

from __future__ import annotations
import argparse, ast, re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import numpy as np
from tqdm import tqdm

# ruBERT (SBERT)
import torch
from sentence_transformers import SentenceTransformer

# --------- Константы путей ---------
RAW_DEFAULT = "../data/raw_data/vacancies_master.csv"
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_CSV = OUT_DIR / "extracted_skills.csv"
EMB_PARQUET = OUT_DIR / "embeddings.parquet"

# --------- Фильтры и нормализация ---------
# стоп-«навыки» (шумовые токены)
STOP_SKILLS: Set[str] = {
    "it",  # частый шум
}

# Разрешаем односимвольный навык "r" (язык R), остальные односимвольные — отбрасываем
def is_valid_skill(token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    if len(t) == 1 and t != "r":
        return False
    if t in STOP_SKILLS:
        return False
    return True

def parse_list_field(x) -> List[str]:
    """Поддержка форматов: уже-список, строка-список '[...]', строка с запятыми."""
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                lst = ast.literal_eval(s)
                return [str(i).strip() for i in lst if str(i).strip()]
            except Exception:
                pass
        # запасной вариант: строка с запятыми
        if "," in s:
            return [i.strip() for i in s.split(",") if i.strip()]
        if s:
            return [s]
    return []

# --------- Извлечение словаря из key_skills ---------
def build_vocab_from_key_skills(df: pd.DataFrame, top_k: int | None = None) -> List[str]:
    """Собираем словарь кандидатов из столбца key_skills (если есть)."""
    if "key_skills" not in df.columns:
        return []
    all_tokens: List[str] = []
    for s in df["key_skills"].fillna(""):
        all_tokens += parse_list_field(s) if isinstance(s, str) and s.startswith("[") else [i.strip() for i in str(s).split(",")]
    normed = [t.lower().strip() for t in all_tokens if is_valid_skill(t)]
    vc = pd.Series(normed).value_counts()
    if top_k:
        vc = vc.head(top_k)
    return vc.index.tolist()

# --------- Извлечение из описаний на основе словаря ---------
def extract_from_descriptions(df: pd.DataFrame, vocab: Iterable[str]) -> List[List[str]]:
    """Грубое извлечение из description по словарю (без rapidfuzz)."""
    vocab = sorted(set([v.strip().lower() for v in vocab if is_valid_skill(v)]), key=len, reverse=True)
    if not vocab:
        return [[] for _ in range(len(df))]

    # соберём regex-паттерн по словарю, экранируя спецсимволы
    # используем \b для однословных и простую подстроку для сочетаний с пробелом
    word_terms = [re.escape(v) for v in vocab if " " not in v]
    phrase_terms = [re.escape(v) for v in vocab if " " in v]
    # \bterm\b только для слов; фразы ищем как подстроки (с учётом пробелов)
    word_pat = r"\b(" + "|".join(word_terms) + r")\b" if word_terms else None
    phrases = [re.compile(p, flags=re.IGNORECASE) for p in phrase_terms]

    out: List[List[str]] = []
    texts = df.get("description", "").fillna("").astype(str).tolist()
    for txt in texts:
        t = txt.lower()
        found: Set[str] = set()
        if word_pat:
            for m in re.finditer(word_pat, t, flags=re.IGNORECASE):
                if is_valid_skill(m.group(1)):
                    found.add(m.group(1).lower())
        for ph in phrases:
            for m in ph.finditer(t):
                s = m.group(0).lower()
                if is_valid_skill(s):
                    found.add(s)
        out.append(sorted(found))
    return out

# --------- Векторизация ---------
def batched_encode(model: SentenceTransformer, texts: Iterable[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    embs = []
    texts = list(texts)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        with torch.inference_mode():
            e = model.encode(batch, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False)
        embs.append(e)
    return np.vstack(embs) if embs else np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

# --------- Основной пайплайн ---------
def run(src: str, model_id: str = "ai-forever/sbert_large_nlu_ru", max_rows: int = 0, batch_size: int = 64):
    # 1) Загрузка
    df = pd.read_csv(src, dtype={"id": str})
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()

    # 2) Подготовка текстов
    df["description"] = df.get("description", "").fillna("").astype(str).str.replace("\n", " ", regex=False).str.strip()

    # 3) Получаем/строим skills_list
    if "skills_list" in df.columns:
        skills_list = df["skills_list"].apply(parse_list_field)
    else:
        # строим словарь по key_skills и извлекаем из описаний
        vocab = build_vocab_from_key_skills(df)
        skills_list = pd.Series(extract_from_descriptions(df, vocab))

    # пост-обработка: фильтр и нормализация
    skills_list = skills_list.apply(lambda lst: sorted({s.lower().strip() for s in lst if is_valid_skill(s)}))

    # 4) Сохраняем таблицу (id, description, skills_list)
    to_save = pd.DataFrame({
        "id": df["id"].astype(str),
        "description": df["description"],
        "skills_list": skills_list.apply(lambda x: list(x))
    })
    to_save.to_csv(EXTRACTED_CSV, index=False, encoding="utf-8-sig")

    # 5) Векторизация (описания + строковое представление навыков)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)

    skills_text = skills_list.apply(lambda lst: " ".join(lst))
    desc_emb = batched_encode(model, to_save["description"], batch_size=batch_size)
    skills_emb = batched_encode(model, skills_text, batch_size=batch_size)

    emb_df = pd.DataFrame({
        "id": to_save["id"],
        "desc_emb": list(map(lambda v: v.astype(float).tolist(), desc_emb)),
        "skills_emb": list(map(lambda v: v.astype(float).tolist(), skills_emb)),
    })
    emb_df.to_parquet(EMB_PARQUET, index=False)
    print(f"✓ Saved: {EXTRACTED_CSV}")
    print(f"✓ Saved: {EMB_PARQUET}")
    print(f"Device: {device}, model: {model_id}, rows: {len(to_save)}")


# --------- CLI ---------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Извлечение и векторизация навыков (ruBERT/SBERT).")
    p.add_argument("--src", type=str, default=RAW_DEFAULT, help="Путь или URL к vacancies_master.csv")
    p.add_argument("--model_id", type=str, default="ai-forever/sbert_large_nlu_ru", help="HF модель для эмбеддингов")
    p.add_argument("--max_rows", type=int, default=0, help="Ограничить число строк (0 — без ограничения)")
    p.add_argument("--batch_size", type=int, default=64, help="Размер батча для эмбеддингов")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(src=args.src, model_id=args.model_id, max_rows=args.max_rows, batch_size=args.batch_size)
