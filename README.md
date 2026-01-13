# Job-Market-Skills-Gap-Analyzer
Job Market Skills Gap Analyzer is a data science project that analyzes job descriptions to identify in-demand skills and gaps between market requirements and candidate profiles. It uses Python, NLP, and data visualization to provide skill match insights and guide targeted upskilling.

# What it does:
- Loads job postings CSV and course catalog CSV (or uses built-in sample data)
- Extracts skills using a simple rule-based extractor + skills dictionary
- Standardizes skills
- Computes demand (from jobs) and supply (from courses, weighted by enrollments)
- Computes a skills gap score and shows top gaps and over-hyped skills
- Runs a Streamlit dashboard for exploration

# How to run:
1) Install dependencies: pip install pandas streamlit plotly
(optional: pip install spacy and download a model if you want more advanced NLP)
2) Run: streamlit run job_skills_gap_analyzer.py

# Notes:
- This is a starter scaffold. Replace sample data with real CSVs or implement scrapers.
- CSV formats expected (if uploading):
jobs.csv -> columns: job_id, title, company, location, description, date_posted
courses.csv -> columns: course_id, title, platform, description, skills (optional), enrollments


"""
