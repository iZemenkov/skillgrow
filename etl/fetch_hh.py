# Загрузка библиотек

import argparse
import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Константы и переменные пути

RAW_DIR = Path("../data/raw_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = RAW_DIR / "vacancies_master.csv"

MASTER_COLUMNS = [
    "id", "name", "area", "experience",
    "key_skills", "description"
]

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "SkillGrow/1.0"})
    s.trust_env = False              # не брать HTTP(S)_PROXY из окружения
    return s

SESSION = get_session()

# Эндпоинт поиска вакансий через API
BASE_URL = "https://api.hh.ru/vacancies"

# Параметры поиска по умолчанию
DEFAULT_PARAMS = {
    "text": "name:(data analyst) OR name:(data scientist)",
    "area": "113",                # Россия
    "per_page": 100,              # вакансий на страницу
    "page": 0,                    # номер страницы
    "only_with_salary": False,    # не только с указанной зарплатой
    "search_field": "name",       # поле для поиска текста
    "date_from": None,
    "date_to": None               # фильтры по дате
}

# функции мастер-файла

def ensure_master(master_path: Path = MASTER_PATH) -> None:
    """Гарантированно создаёт пустой мастер-CSV с нужными колонками."""
    master_path.parent.mkdir(parents=True, exist_ok=True)
    if not master_path.exists():
        pd.DataFrame(columns=MASTER_COLUMNS).to_csv(
            master_path, index=False, encoding="utf-8-sig"
        )

def get_master_ids(master_path: Path = MASTER_PATH) -> set:
    """Читает множество уже известных id из мастера."""
    ensure_master(master_path)
    if master_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(master_path, dtype=str)
    if "id" not in df.columns:
        return set()
    return set(df["id"].dropna().astype(str))

# подготовка списка вакансий

def fetch_basic_vacancies(params: Dict[str, Any],
                          max_pages: Optional[int] = None,
                          pause: float = 1.75) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками: id, name, area, experience.
    Ходит по /vacancies (поиск), обходит страницы.
    """

    # первичный запрос узнаёт количество страниц
    r = SESSION.get(BASE_URL, params=params,timeout=30)
    r.raise_for_status()
    data = r.json()
    pages_total = data.get("pages", 0)

    if max_pages is not None:
        pages_total = min(pages_total, max_pages)

    rows = []
    for page in range(pages_total):
        page_params = {**params, "page": page}
        resp = SESSION.get(BASE_URL, params=page_params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        for it in items:
            rows.append({
                "id": it.get("id"),
                "name": it.get("name"),
                "area": (it.get("area") or {}).get("name"),
                "experience": (it.get("experience") or {}).get("name"),
            })

        time.sleep(pause)  # бережём лимиты

    return pd.DataFrame(rows, columns=["id", "name", "area", "experience"])

def filter_new_ids(df_basic: pd.DataFrame, master_ids: set) -> pd.DataFrame:
    """Оставляет только строки, id которых отсутствуют в мастере."""
    if df_basic.empty:
        return df_basic
    df_basic = df_basic.astype({"id": str})
    return df_basic[~df_basic["id"].isin(master_ids)].drop_duplicates(subset=["id"])

# Детали для новых вакансий

def fetch_vacancy_details(ids: Iterable[str],
                          pause: float = 1.75,
                          base_url: str = "https://api.hh.ru/vacancies") -> pd.DataFrame:
    """
    По списку id ходит в /vacancies/{id}, забирает key_skills и description.
    """
    rows = []
    for vid in ids:
        r = SESSION.get(f"{base_url}/{vid}", timeout=30)
        if r.status_code != 200:
            time.sleep(pause)
            continue

        v = r.json()
        ks = ", ".join([k.get("name") for k in (v.get("key_skills") or [])])

        desc_html = (v.get("description") or "")
        description = BeautifulSoup(desc_html, "html.parser").get_text().strip()

        rows.append({
            "id": str(v.get("id")),
            "key_skills": ks,
            "description": description
        })

        time.sleep(pause)

    return pd.DataFrame(rows, columns=["id", "key_skills", "description"])

def build_new_dataset(df_new_basic: pd.DataFrame) -> pd.DataFrame:
    """
    Соединяет basic-строки с деталями для новых id.
    """
    if df_new_basic.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    new_ids = df_new_basic["id"].astype(str).tolist()
    df_details = fetch_vacancy_details(new_ids, pause=1.75)

    if df_details.empty:
        df_new_full = df_new_basic.copy()
        df_new_full["key_skills"] = ""
        df_new_full["description"] = ""
        return df_new_full[MASTER_COLUMNS]

    df_new_full = (
        df_new_basic.astype({"id": str})
        .merge(df_details, on="id", how="left")
    )
    return df_new_full[MASTER_COLUMNS]

# Сохранение новых вакансий и добавление в мастер-файл

def save_new_csv(df_new_full: pd.DataFrame,
                 out_dir: Path | str = RAW_DIR,
                 prefix: str = "vacancies_new") -> str:
    """Сохраняет новую партию CSV с таймстампом в RAW_DIR."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.today().strftime('%Y%m%d')
    fpath = out_dir / f"{prefix}_{ts}.csv"
    df_new_full.to_csv(fpath, index=False, encoding="utf-8-sig")
    return str(fpath)


def append_to_master(df_new_full: pd.DataFrame,
                     master_path: Path = MASTER_PATH) -> int:
    """Добавляет новые строки в мастер, удаляя дубликаты по id."""
    ensure_master(master_path)
    if df_new_full.empty:
        return 0

    if master_path.stat().st_size > 0:
        df_master = pd.read_csv(master_path, dtype=str)
    else:
        df_master = pd.DataFrame(columns=MASTER_COLUMNS)

    before = len(df_master)
    df_updated = (
        pd.concat([df_master, df_new_full.astype(str)], ignore_index=True)
        .drop_duplicates(subset=["id"])
    )
    df_updated.to_csv(master_path, index=False, encoding="utf-8-sig")
    added = len(df_updated) - before
    return added

# Полный сценарий

def fetch_and_upsert(params: Dict[str, Any],
                     max_pages: int = 5,
                     pause: float = 1.75,
                     save_prefix: str = "vacancies_new",
                     master_path: Path = MASTER_PATH) -> Dict[str, Any]:
    """
    1) ensure_master + get_master_ids
    2) fetch_basic_vacancies -> filter_new_ids
    3) build_new_dataset -> save_new_csv -> append_to_master
    """
    ensure_master(master_path)
    master_ids = get_master_ids(master_path)

    df_basic = fetch_basic_vacancies(params, max_pages=max_pages, pause=pause)
    df_new_basic = filter_new_ids(df_basic, master_ids)

    result = {
        "searched_total": int(len(df_basic)),
        "new_ids": int(len(df_new_basic)),
        "new_csv_path": None,
        "added_to_master": 0,
        "master_path": str(master_path)
    }

    if df_new_basic.empty:
        return result

    df_new_full = build_new_dataset(df_new_basic)
    new_csv_path = save_new_csv(df_new_full, out_dir=RAW_DIR, prefix=save_prefix)
    added = append_to_master(df_new_full, master_path=master_path)

    result.update({
        "new_csv_path": new_csv_path,
        "added_to_master": int(added)
    })
    return result

# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Инкрементальная выгрузка вакансий HH и обновление мастер-CSV."
    )
    p.add_argument("--text", type=str,
                   default=DEFAULT_PARAMS["text"],
                   help="Поисковая строка HH (поддерживает поля name:, логические операторы)")
    p.add_argument("--area", type=str,
                   default=DEFAULT_PARAMS["area"],
                   help="ID региона (113 — Россия, 1 — Москва, 2 — СПб)")
    p.add_argument("--per_page", type=int,
                   default=DEFAULT_PARAMS["per_page"],
                   help="Вакансий на страницу (макс. 100)")
    p.add_argument("--search_field", type=str,
                   default=DEFAULT_PARAMS["search_field"],
                   help="Поле поиска: name/description/etc")
    p.add_argument("--max_pages", type=int, default=5,
                   help="Сколько страниц пройти (ограничение поверх API)")
    p.add_argument("--pause", type=float, default=1.75,
                   help="Пауза между запросами (сек)")
    p.add_argument("--date_from", type=str, default=None,
                   help="ISO-8601, напр. 2025-10-01T00:00:00")
    p.add_argument("--date_to", type=str, default=None,
                   help="ISO-8601, напр. 2025-11-01T00:00:00")
    p.add_argument("--only_with_salary", action="store_true",
                   help="Только вакансии с указанной зарплатой")
    p.add_argument("--save_prefix", type=str, default="vacancies_new",
                   help="Префикс для CSV-партии новых")
    p.add_argument("--master_path", type=str, default=str(MASTER_PATH),
                   help="Путь к мастер-CSV (по умолчанию ../data/raw_data/vacancies_master.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    params = {
        "text": args.text,
        "area": args.area,
        "per_page": args.per_page,
        "page": 0,
        "only_with_salary": bool(args.only_with_salary),
        "search_field": args.search_field,
        "date_from": args.date_from,
        "date_to": args.date_to
    }

    summary = fetch_and_upsert(
        params=params,
        max_pages=args.max_pages,
        pause=args.pause,
        save_prefix=args.save_prefix,
        master_path=Path(args.master_path)
    )

    print(
        f"searched_total={summary['searched_total']} | "
        f"new_ids={summary['new_ids']} | "
        f"added_to_master={summary['added_to_master']} | "
        f"batch_csv={summary['new_csv_path']} | "
        f"master={summary['master_path']}"
    )


if __name__ == "__main__":
    main()














