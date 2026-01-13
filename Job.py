import streamlit as st
import pandas as pd
import re
from collections import Counter, defaultdict
import math
import plotly.express as px

# -------------------------
# CONFIG / SKILLS DICTIONARY
# -------------------------
# A small canonical skills list to map raw tokens to canonical skills.
# Expand this mapping with domain-specific terms for better results.
SKILLS_DICT = {
    'python': 'Python',
    'sql': 'SQL',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'machine learning': 'Machine Learning',
    'ml': 'Machine Learning',
    'deep learning': 'Deep Learning',
    'tensorflow': 'TensorFlow',
    'pytorch': 'PyTorch',
    'aws': 'AWS',
    'azure': 'Azure',
    'gcp': 'GCP',
    'kubernetes': 'Kubernetes',
    'docker': 'Docker',
    'spark': 'Apache Spark',
    'hadoop': 'Hadoop',
    'nlp': 'NLP',
    'natural language processing': 'NLP',
    'sql server': 'SQL',
    'power bi': 'Power BI',
    'tableau': 'Tableau',
    'react': 'React',
    'node': 'Node.js',
    'javascript': 'JavaScript',
}

# common multi-word phrases we want to detect first
PHRASES = sorted([k for k in SKILLS_DICT.keys() if ' ' in k], key=lambda x: -len(x))

# -------------------------
# HELPERS: Skill extraction
# -------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    # replace punctuation with spaces
    text = re.sub(r"[\n\r\/\\,;:\(\)\[\]\-\.\"]", ' ', text)
    return text


def extract_skills_from_text(text: str, extra_skill_patterns=None):
    """Simple rule-based skill extractor using SKILLS_DICT keys and regex.
    Returns a list of canonical skill names.
    """
    text_norm = normalize_text(text)
    found = []

    # detect phrase matches first (e.g., 'machine learning')
    for phrase in PHRASES:
        if phrase in text_norm:
            found.append(SKILLS_DICT[phrase])
            # remove the phrase to avoid double counting
            text_norm = text_norm.replace(phrase, ' ')

    # token-based detection
    tokens = re.findall(r"\b[a-z0-9+#._-]+\b", text_norm)
    for tok in tokens:
        if tok in SKILLS_DICT:
            found.append(SKILLS_DICT[tok])

    # custom extra patterns (list of regex) — optional
    if extra_skill_patterns:
        for pat, label in extra_skill_patterns:
            if re.search(pat, text, flags=re.I):
                found.append(label)

    # deduplicate preserving order
    seen = set()
    uniq = []
    for s in found:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

# -------------------------
# SAMPLE DATA (for quick demo)
# -------------------------

SAMPLE_JOBS = [
    {
        'job_id': 'J001',
        'title': 'Data Scientist',
        'company': 'Acme AI',
        'location': 'Bengaluru, India',
        'description': 'Looking for Data Scientist with strong Python, SQL, Pandas experience. Familiarity with TensorFlow or PyTorch is a plus. Knowledge of AWS and Docker preferred.',
        'date_posted': '2025-09-10'
    },
    {
        'job_id': 'J002',
        'title': 'Cloud Engineer',
        'company': 'CloudCorp',
        'location': 'Remote',
        'description': 'Experience with AWS, Kubernetes, Docker, and Terraform. Knowledge of GCP is a plus.',
        'date_posted': '2025-09-12'
    },
    {
        'job_id': 'J003',
        'title': 'Business Intelligence Analyst',
        'company': 'RetailCo',
        'location': 'Mumbai, India',
        'description': 'Power BI, SQL, and data visualization skills required. Python or R is helpful.',
        'date_posted': '2025-09-05'
    }
]

SAMPLE_COURSES = [
    {
        'course_id': 'C001',
        'title': 'Complete Python Bootcamp',
        'platform': 'Coursera',
        'description': 'Learn Python, NumPy, Pandas for data analysis.',
        'skills': 'Python, NumPy, Pandas',
        'enrollments': 1200000
    },
    {
        'course_id': 'C002',
        'title': 'Kubernetes Masterclass',
        'platform': 'Udemy',
        'description': 'Hands-on Kubernetes, Docker, and container orchestration.',
        'skills': 'Kubernetes, Docker',
        'enrollments': 45000
    },
    {
        'course_id': 'C003',
        'title': 'Intro to SQL',
        'platform': 'edX',
        'description': 'SQL fundamentals for querying databases.',
        'skills': 'SQL',
        'enrollments': 300000
    }
]

# -------------------------
# CORE: build frequency tables
# -------------------------

def build_skill_counts_from_jobs(jobs_df: pd.DataFrame):
    all_skills = []
    for _, row in jobs_df.iterrows():
        text = f"{row.get('title','')} {row.get('description','')}"
        skills = extract_skills_from_text(text)
        all_skills.extend(skills)
    counts = Counter(all_skills)
    return pd.DataFrame(counts.items(), columns=['skill', 'demand_count']).sort_values('demand_count', ascending=False)


def build_skill_supply_from_courses(courses_df: pd.DataFrame):
    # we weight course skills by enrollments (if available)
    supply = defaultdict(float)
    for _, row in courses_df.iterrows():
        text = f"{row.get('title','')} {row.get('description','')} {row.get('skills','') or ''}"
        skills = extract_skills_from_text(text)
        weight = float(row.get('enrollments', 1) or 1)
        for s in skills:
            supply[s] += weight
    # convert to counts — normalize by dividing by 1k to keep numbers readable
    supply_items = [(k, v) for k, v in supply.items()]
    return pd.DataFrame(supply_items, columns=['skill', 'supply_score']).sort_values('supply_score', ascending=False)


def merge_demand_supply(demand_df: pd.DataFrame, supply_df: pd.DataFrame):
    df = demand_df.merge(supply_df, on='skill', how='outer').fillna(0)
    # normalize columns to 0..1 for fair comparison
    if df['demand_count'].max() > 0:
        df['demand_norm'] = df['demand_count'] / df['demand_count'].max()
    else:
        df['demand_norm'] = 0
    if df['supply_score'].max() > 0:
        df['supply_norm'] = df['supply_score'] / df['supply_score'].max()
    else:
        df['supply_norm'] = 0
    # gap score: demand_norm - supply_norm (positive -> gap)
    df['gap_score'] = df['demand_norm'] - df['supply_norm']
    df = df.sort_values('gap_score', ascending=False)
    return df

# -------------------------
# STREAMLIT APP
# -------------------------

def run_app():
    st.set_page_config(layout='wide', page_title='Job Market Skills Gap Analyzer')
    st.title('Job Market Skills Gap Analyzer')
    st.markdown('Compare **job demand** (from job postings) vs **course supply** (from online courses) to find skills gaps and over-hyped skills.')

    st.sidebar.header('Data Inputs')
    use_sample = st.sidebar.checkbox('Use sample data (quick demo)', value=True)

    if use_sample:
        jobs_df = pd.DataFrame(SAMPLE_JOBS)
        courses_df = pd.DataFrame(SAMPLE_COURSES)
    else:
        jobs_file = st.sidebar.file_uploader('Upload jobs CSV', type=['csv'])
        courses_file = st.sidebar.file_uploader('Upload courses CSV', type=['csv'])
        if jobs_file is not None:
            jobs_df = pd.read_csv(jobs_file)
        else:
            st.sidebar.warning('Please upload jobs CSV or enable sample data')
            jobs_df = pd.DataFrame(SAMPLE_JOBS)
        if courses_file is not None:
            courses_df = pd.read_csv(courses_file)
        else:
            st.sidebar.info('No courses CSV uploaded — using sample courses')
            courses_df = pd.DataFrame(SAMPLE_COURSES)

    # Basic preview
    st.subheader('Preview data')
    c1, c2 = st.columns(2)
    with c1:
        st.write('Jobs (sample)')
        st.dataframe(jobs_df.head(10))
    with c2:
        st.write('Courses (sample)')
        st.dataframe(courses_df.head(10))

    # Build counts
    demand_df = build_skill_counts_from_jobs(jobs_df)
    supply_df = build_skill_supply_from_courses(courses_df)

    merged = merge_demand_supply(demand_df, supply_df)

    st.subheader('Top skills by demand')
    if not demand_df.empty:
        fig = px.bar(demand_df.head(20), x='skill', y='demand_count', title='Top demanded skills')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Top skills by supply (weighted by enrollments)')
    if not supply_df.empty:
        fig2 = px.bar(supply_df.head(20), x='skill', y='supply_score', title='Top supplied skills (courses)')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Skills Gap (demand_norm - supply_norm)')
    st.write('Positive gap_score = skill is more demanded than supplied (opportunity). Negative = oversupplied/over-hyped.')
    st.dataframe(merged[['skill', 'demand_count', 'supply_score', 'demand_norm', 'supply_norm', 'gap_score']].head(50))

    c3, c4 = st.columns(2)
    with c3:
        st.write('Top skills gap (opportunities)')
        if not merged.empty:
            top_gap = merged[merged['gap_score']>0].head(10)
            fig3 = px.bar(top_gap, x='skill', y='gap_score', title='Top skills gaps')
            st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.write('Top over-hyped skills (oversupplied)')
        if not merged.empty:
            over = merged.sort_values('gap_score', ascending=True).head(10)
            fig4 = px.bar(over, x='skill', y='gap_score', title='Top over-hyped skills (negative gap)')
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown('---')
    st.subheader('Export results')
    if st.button('Download skills gap CSV'):
        csv = merged.to_csv(index=False)
        st.download_button('Click to download CSV', data=csv, file_name='skills_gap.csv', mime='text/csv')

    st.markdown('**Next steps / improvements:**')
    st.write('- Replace rule-based skill extractor with spaCy + custom NER or use existing skills taxonomy (O*NET, ESCO).')
    st.write('- Scrape live job boards and learning platforms (observe ToS!).')
    st.write('- Add time-series snapshots to track rising skills.')
    st.write('- Cluster skills into domains and show regional comparisons.')


if __name__ == '__main__':
    run_app()
