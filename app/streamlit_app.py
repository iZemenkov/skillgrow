import streamlit as st
import pandas as pd
import numpy as np
import docx
import re
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Импортируем только константы для демонстрации (никакого реального вызова API)
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from etl.fetch_hh import DEFAULT_PARAMS as HH_DEFAULT_PARAMS


st.set_page_config(
    page_title="SkillGrow — рекомендации по навыкам",
    layout="wide",
)

# ------------------------------
# Нормализация навыков и синонимы
# ------------------------------
def normalize_skill(s: str) -> str:
    s = s.strip().lower()

    mapping = {
        # ML_DL
        "ml": "machine learning",
        "машинное обучение": "machine learning",
        "классическое машинное обучение": "machine learning",
        "dl": "deep learning",
        "deep learning": "dl",
        "nlp": "natural language processing",
        "nlp": "llm",

        # Data Science / DS
        "ds": "data science",
        "data scientist": "data science",
        "дата саентист": "data science",
        "дата сайнтист": "data science",

        # БД
        "postgres": "postgresql",
        "postgre": "postgresql",

        # BI / визуализация
        "bi": "business intelligence",
        "powerbi": "power bi",
        "дашборды":"bi"
    }

    return mapping.get(s, s)


# ------------------------------
# Загрузка обработанных данных
# ------------------------------
MASTER_URL = "https://izemenkov.github.io/skillgrow/data/raw_data/vacancies_master.csv"
EXTRACTED_CSV = "https://izemenkov.github.io/skillgrow/data/processed/extracted_skills.csv"
EMB_PARQUET = "https://izemenkov.github.io/skillgrow/data/processed/embeddings.parquet"



master_df = pd.read_csv(MASTER_URL)
skills_df = pd.read_csv(EXTRACTED_CSV)
emb_df = pd.read_parquet(EMB_PARQUET)

desc_emb = np.vstack(emb_df["desc_emb"].values)
skills_emb = np.vstack(emb_df["skills_emb"].values)

# Словарь навыков из вакансий (для поиска в резюме)
all_skills_raw = []
for lst in skills_df["skills_list"].apply(eval):
    for sk in lst:
        all_skills_raw.append(sk)

all_skills = sorted({s for s in all_skills_raw})


# ------------------------------
# Парсинг резюме
# ------------------------------
def load_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def clean_text(t: str) -> str:
    t = t.lower()
    return re.sub(r"[^a-zа-я0-9+/.,\- ]", " ", t)


def extract_resume_skills(text: str, vocab) -> list[str]:
    """
    Простейшее извлечение навыков из резюме:
    по словарю навыков из вакансий.
    """
    found = []
    for sk in vocab:
        sk_l = sk.lower().strip()
        if not sk_l:
            continue
        if re.search(rf"\b{re.escape(sk_l)}\b", text):
            found.append(sk_l)

    normalized = {normalize_skill(s) for s in found}
    return sorted(normalized)


# ------------------------------
# Модель эмбеддингов
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("ai-forever/sbert_large_nlu_ru")


model = load_model()


def embed(text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True)[0]


# ------------------------------
# Рекомендательная функция
# ------------------------------
def recommend_skills(
    resume_text: str,
    resume_emb: np.ndarray,
    resume_skills: list[str],
    similar_top_n: int = 20,
) -> pd.DataFrame:
    """
    1) Находим топ-N похожих вакансий по описанию.
    2) Считаем, какие навыки в них встречаются чаще всего.
    3) Для каждого такого навыка считаем семантическую близость к резюме.
    4) Отфильтровываем навыки, которые уже есть в резюме (с учётом синонимов).
    """

    resume_skills_norm = {normalize_skill(s) for s in resume_skills}

    sim_desc = cosine_similarity([resume_emb], desc_emb)[0]
    top_idx = np.argsort(sim_desc)[::-1][:similar_top_n]

    similar_vacancies = skills_df.iloc[top_idx]

    freq: dict[str, int] = {}
    for lst in similar_vacancies["skills_list"]:
        for raw_sk in eval(lst):
            canon = normalize_skill(raw_sk)
            if not canon:
                continue
            freq[canon] = freq.get(canon, 0) + 1

    if not freq:
        return pd.DataFrame(columns=["skill", "freq", "similarity"])

    freq_df = (
        pd.DataFrame([{"skill": k, "freq": v} for k, v in freq.items()])
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )

    unique_skills = freq_df["skill"].tolist()
    skill_texts = [s for s in unique_skills]
    skill_embs = model.encode(skill_texts, normalize_embeddings=True)
    sim = cosine_similarity([resume_emb], skill_embs)[0]
    freq_df["similarity"] = sim

    def is_covered(skill: str) -> bool:
        canon = normalize_skill(skill)
        return canon in resume_skills_norm

    freq_df = freq_df[~freq_df["skill"].apply(is_covered)]

    freq_df = freq_df.sort_values(
        ["freq", "similarity"], ascending=[False, False]
    ).reset_index(drop=True)

    return freq_df, similar_vacancies


# ------------------------------
# Визуализация Plotly
# ------------------------------
def plot_skill_recommendations_plotly(rec_df: pd.DataFrame, top_n=20):
    if rec_df.empty:
        return None

    df = (
        rec_df
        .copy()
        .sort_values(["freq", "similarity"], ascending=[False, False])
        .head(top_n)
    )

    fig = px.scatter(
        df,
        x="similarity",
        y="skill",
        size="freq",
        color="similarity",
        size_max=30,
        color_continuous_scale="Viridis",
        labels={
            "similarity": "Семантическая близость к резюме",
            "freq": "Частота в похожих вакансиях",
            "skill": "Навык",
        },
        title="Рекомендованные навыки для развития",
        hover_data={
            "freq": True,
            "similarity": ":.3f",
            "skill": False,
        },
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=df.sort_values(
            ["freq", "similarity"], ascending=[True, True]
        )["skill"].tolist()
    )

    fig.update_layout(
        xaxis=dict(
            range=[
                df["similarity"].min() - 0.02,
                df["similarity"].max() + 0.02,
            ]
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=150, r=40, t=60, b=60),
    )

    return fig


# ------------------------------
# UI: вкладки
# ------------------------------
st.title("SkillGrow — рекомендации по развитию навыков для Data Science")

tabs = st.tabs([
    "1️⃣ Анализ резюме",
    "2️⃣ Как работает сервис"
])

# ==============================
# Вкладка 1 — Анализ резюме
# ==============================
with tabs[0]:
    st.subheader("Анализ резюме и рекомендации по навыкам")

    uploaded = st.file_uploader("Загрузите ваше резюме (DOCX)", type=["docx"])

    if uploaded:
        text = load_docx(uploaded)
        clean = clean_text(text)

        resume_emb = embed(clean)
        resume_sk = extract_resume_skills(clean, all_skills)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Извлечённый текст резюме**")
            st.write(text)

        with col2:
            st.markdown("**Найденные навыки (нормализованные)**")
            if resume_sk:
                st.write(", ".join(sorted(resume_sk)))
            else:
                st.write("Навыков не найдено (по словарю из вакансий).")

        st.sidebar.header("Настройки рекомендаций")
        top_n = st.sidebar.slider(
            "Сколько навыков показывать на графике", 5, 30, 15
        )
        similar_top_n = st.sidebar.slider(
            "Сколько похожих вакансий учитывать", 5, 50, 20
        )

        rec_df, similar_vacancies = recommend_skills(
            clean,
            resume_emb,
            resume_skills=resume_sk,
            similar_top_n=similar_top_n,
        )


        similar_vacancies['skills_list'] = similar_vacancies['skills_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        similar_vacancies['Недостающие навыки'] = similar_vacancies['skills_list'].apply(lambda lst: [x for x in lst if x not in resume_sk]).apply(lambda lst: ", ".join(lst))
        similar_vacancies['Имеющиеся навыки'] = similar_vacancies['skills_list'].apply(lambda lst: [x for x in lst if x in resume_sk]).apply(lambda lst: ", ".join(lst))
        similar_vacancies['skills_list'] = similar_vacancies['skills_list'].apply(lambda lst: ", ".join(lst))


        similar_vacancies = similar_vacancies.rename(
            columns={'id':'id',
                     'description':'Описание вакансии',
                     'skills_list':'Навыки вакансии'}
        )


        st.markdown("### Список приоритетных вакансий")
        st.dataframe(similar_vacancies[['id','Навыки вакансии', 'Имеющиеся навыки','Недостающие навыки','Описание вакансии']])

        st.markdown("### График приоритетных навыков")
        st.write(
            "Рекомендация не означает, что у вас точно нет навыка — "
            "она показывает, какие навыки чаще всего встречаются в похожих "
            "вакансиях и не упомянуты в вашем резюме."
        )

        fig = plot_skill_recommendations_plotly(rec_df, top_n=top_n)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет рекомендаций для отображения.")

        rec_view = rec_df.rename(
            columns={
                "skill": "Навык",
                "freq": "Частота в похожих вакансиях",
                "similarity": "Семантическая близость к резюме",
            }
        )
        st.markdown("### Таблица рекомендаций")
        st.dataframe(rec_view.head(top_n))
    else:
        st.info("Загрузите файл резюме, чтобы получить рекомендации.")

# ==============================
# Вкладка 2 — Описание работы сервиса
# ==============================
with tabs[1]:
    st.subheader("Как собираются вакансии с hh.ru")

    st.markdown(
        """
        **Пайплайн сбора данных** (скрипт `etl/fetch_hh.py`):

        1. Формируется запрос к API hh.ru со строкой поиска по позициям Data Analyst / Data Scientist / специалист по машинному обучению.
        2. Проходим по страницам выдачи (`/vacancies`) и собираем базовую информацию:
           `id`, `name`, `area`, `experience`.
        3. Для новых `id` ходим в `/vacancies/{id}` и забираем:
           `key_skills`, `description`.
        4. Сохраняем партию новых вакансий в `data/raw_data/vacancies_new_YYYYMMDD.csv`.
        5. Обновляем мастер-файл `vacancies_master.csv` без дублей по `id`.
        """
    )

    st.markdown("**Пример параметров запроса к API hh.ru:**")
    st.json(HH_DEFAULT_PARAMS)

    st.markdown("**Пример мастер-файла (сырые вакансии с hh.ru):**")
    st.dataframe(
        master_df.head(),
        use_container_width=True
    )

    st.code(
        """
from etl.fetch_hh import fetch_and_upsert, DEFAULT_PARAMS

summary = fetch_and_upsert(
    params=DEFAULT_PARAMS,
    max_pages=5,
    pause=1.75,
)
print(summary)
        """,
        language="python",
    )

    st.subheader("Как извлекаются навыки и строятся эмбеддинги")

    st.markdown(
        """
        **Пайплайн обработки** (скрипт `processing/skill_extraction.py`):

        1. Берём `vacancies_master.csv` и очищаем тексты описаний.
        2. Формируем словарь навыков на основе `key_skills`.
        3. Извлекаем навыки из `description` по словарю, приводим к нижнему регистру,
           фильтруем шум (`it`, одиночные буквы, кроме `r` и т.п.).
        4. Сохраняем таблицу `extracted_skills.csv` с колонками:
           `id`, `description`, `skills_list`.
        5. Строим эмбеддинги ruBERT (модель `ai-forever/sbert_large_nlu_ru`) для:
           – текста описания вакансии (`desc_emb`),  
           – строкового представления навыков (`skills_emb`).  
        6. Сохраняем эмбеддинги в `embeddings.parquet`.
        """
    )



    st.markdown("**Пример извлечённых навыков из вакансий:**")
    st.write(skills_df.head())
    st.markdown("**Пример векторизованного представления описаний вакансий и навыков:**")
    st.write(emb_df.head())

    st.markdown("**Размеры векторных представлений:**")
    st.write(f"Вакансий в базе: **{len(skills_df)}**")
    st.write(f"Матрица описаний: `desc_emb.shape = {desc_emb.shape}`")
    st.write(f"Матрица навыков: `skills_emb.shape = {skills_emb.shape}`")

    st.code(
        """
from sentence_transformers import SentenceTransformer
from processing.skill_extraction import batched_encode  # логика батчевой векторизации

model = SentenceTransformer("ai-forever/sbert_large_nlu_ru", device="cuda")
desc_emb = batched_encode(model, df["description"])
skills_emb = batched_encode(model, df["skills_text"])
        """,
        language="python",
    )

    st.info(
        "Эти эмбеддинги используются на первой вкладке, чтобы находить "
        "вакансии, похожие на ваше резюме, и извлекать недостающие навыки."
    )
