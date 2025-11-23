import streamlit as st
import pandas as pd
import numpy as np
import docx
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ------------------------------
# Нормализация навыков и синонимы
# ------------------------------
def normalize_skill(s: str) -> str:
    """Приводим разные написания одного навыка к каноническому виду."""
    s = s.strip().lower()

    mapping = {
        # ML
        "ml": "machine learning",
        "машинное обучение": "machine learning",
        "классическое машинное обучение": "machine learning",

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

        # Прочее — можно расширять по мере надобности
    }

    return mapping.get(s, s)


# ------------------------------
# Загрузка обработанных данных
# ------------------------------
EXTRACTED_CSV = "https://izemenkov.github.io/skillgrow/data/processed/extracted_skills.csv"
EMB_PARQUET = "https://izemenkov.github.io/skillgrow/data/processed/embeddings.parquet"

skills_df = pd.read_csv(EXTRACTED_CSV)
emb_df = pd.read_parquet(EMB_PARQUET)

desc_emb = np.vstack(emb_df["desc_emb"].values)
skills_emb = np.vstack(emb_df["skills_emb"].values)

# Словарь навыков из вакансий
all_skills_raw = []
for lst in skills_df["skills_list"].apply(eval):
    for sk in lst:
        all_skills_raw.append(sk)

# можно использовать "сырые" навыки для поиска по тексту резюме
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
        # ищем точное вхождение по словам
        if re.search(rf"\b{re.escape(sk_l)}\b", text):
            found.append(sk_l)

    # нормализуем (склеиваем синонимы)
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

    # Нормализованный набор навыков из резюме
    resume_skills_norm = {normalize_skill(s) for s in resume_skills}

    # 1) Схожесть описания резюме с описаниями вакансий
    sim_desc = cosine_similarity([resume_emb], desc_emb)[0]
    top_idx = np.argsort(sim_desc)[::-1][:similar_top_n]

    similar_vacancies = skills_df.iloc[top_idx]

    # 2) Частоты навыков в похожих вакансиях (на канонических названиях)
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

    # 3) Семантическая близость навыка к резюме
    unique_skills = freq_df["skill"].tolist()
    skill_texts = [s for s in unique_skills]
    skill_embs = model.encode(skill_texts, normalize_embeddings=True)
    sim = cosine_similarity([resume_emb], skill_embs)[0]

    freq_df["similarity"] = sim

    # 4) Фильтрация навыков, которые уже покрыты в резюме
    def is_covered(skill: str) -> bool:
        canon = normalize_skill(skill)
        return canon in resume_skills_norm

    freq_df = freq_df[~freq_df["skill"].apply(is_covered)]

    # 5) Финальная сортировка: сначала по частоте, затем по семантической близости
    freq_df = freq_df.sort_values(["freq", "similarity"], ascending=[False, False]).reset_index(drop=True)

    return freq_df


# ------------------------------
# Визуализация Plotly
# ------------------------------
def plot_skill_recommendations_plotly(rec_df:pd.DataFrame, top_n=20):
    """
    Красивый интерактивный график рекомендаций:
    - по оси Y — навыки
    - по оси X — семантическая близость к резюме
    - размер пузырька — частота навыка в похожих вакансиях
    - цвет — тоже похожесть (similarity)
    """
    if rec_df.empty:
        print("Нет рекомендаций для отображения.")
        return

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
            "skill": False,  # skill и так по оси Y
        },
    )

    # Чтобы самые важные навыки были сверху
    fig.update_yaxes(categoryorder="array", categoryarray=df.sort_values(
        ["freq", "similarity"], ascending=[True, True]
    )["skill"].tolist())

    fig.update_layout(
        xaxis=dict(range=[df["similarity"].min() - 0.02,
                          df["similarity"].max() + 0.02]),
        template="plotly_white",
        height=600,
        margin=dict(l=150, r=40, t=60, b=60),
    )

    return fig


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Система рекомендаций по улучшению резюме для специалистов Data Science")

st.subheader('Принцип работы')
st.text('Работа системы основана на данных актуальных вакансий с hh.ru. База вакансий обновляется ежедневно.')
st.text('В процессе анализа ваше резюме сравнивается c наиболее близкими по описанию вакансиями, извлеченные навыки сравниваются с навыками, которые чаще других указывает рынок в близких вакансиях.')
st.text('По результатам анализа рекомендуется список навыков, которых не хватает вашему резюме для соответствия указанным вакансиям.')

uploaded = st.file_uploader("Загрузите ваше резюме (DOCX)", type=["docx"])

if uploaded:
    text = load_docx(uploaded)
    clean = clean_text(text)

    # Эмбеддинг резюме
    resume_emb = embed(clean)


    col1, col2 = st.columns(2)
    with col1:

        st.subheader("Извлечённый текст резюме")
        st.write(text)

    with col2:
        # Навыки из резюме
        resume_sk = extract_resume_skills(clean, all_skills)
        st.subheader("Найденные навыки в резюме (нормализованные)")
        if resume_sk:
            st.write(", ".join(sorted(resume_sk)))
        else:
            st.write("Навыков не найдено (по словарю из вакансий)")

    st.sidebar.header("Настройки рекомендаций")
    top_n = st.sidebar.slider("Сколько навыков показывать на графике", 5, 30, 15)
    similar_top_n = st.sidebar.slider("Сколько похожих вакансий учитывать", 5, 50, 20)



    # Рекомендации
    rec_df = recommend_skills(clean, resume_emb, similar_top_n=similar_top_n, resume_skills=resume_sk)



    # График
    st.subheader("График приоритетных навыков")
    st.write('Рекомендация не обозначает что вы не имеете указанных навыков, но отсутствие их в Вашем резюме может повлиять на прохождение им фильтра кандиадтов')
    fig = plot_skill_recommendations_plotly(rec_df, top_n=top_n)
    st.plotly_chart(fig, width='content')


    rec_df = rec_df.rename(columns={"skill": "Навыки", 'freq': 'Частота упоминания в топе вакансий', 'similarity': 'Семантическая близость к резюме'})
    st.subheader("Рекомендации по развитию/указанию навыков")
    st.dataframe(rec_df.head(top_n))
else:
    st.info("Загрузите файл резюме, чтобы получить рекомендации.")

