import os
import smtplib
from typing import List, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from dotenv import load_dotenv
from jobspy import scrape_jobs

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# Configuration
RESUME = os.getenv("RESUME_TEXT", "")
SEARCH_TERM = "Software Engineer"
LOCATION = "Tokyo, Japan"
RESULT_LIMIT = 10
HOURS_OLD = 24
PROXY_URL = os.getenv("PROXY_URL", None)

# Define the output data structure from AI
class JobEvaluation(BaseModel):
    """
    Structure for job evaluation output.
    """

    score: int = Field(description="A relevance score from 0 to 100 based on the resume match and job preferences.")
    reason: str = Field(description="A concise, one-sentence reason for the score.")


# AI model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
)


# structure output
structured_llm = llm.with_structured_output(JobEvaluation)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert tech career coach. Your goal is to evaluate how well a job description matches a candidate's resume."),
    ("user", """
    RESUME (Truncated):
    {resume}

    JOB TITLE: {title}
    JOB DESCRIPTION (Truncated):
    {description}

    Analyze the match. Be strict. If the tech stack is completely different, give a low score.
    """)
])

# Chain
evaluation_chain = prompt_template | structured_llm

# scrape jobs
def get_jobs_data():
    """
    Scrape job listings by JobSpy.
    """
    proxies = [PROXY_URL] if PROXY_URL else None
    print(f"🕵️  CareerScout is searching for '{SEARCH_TERM}' in '{LOCATION}'...")
    print(f"🔌  Proxy: {proxies[0] if proxies else 'None'}")

    try:
        jobs = scrape_jobs(
            site_name=["linkedin"],
            search_term=SEARCH_TERM,
            location=LOCATION,
            result_wanted=RESULT_LIMIT,
            hours_old=HOURS_OLD,
            proxies=proxies
        )

        print(f"✅  Scraped {len(jobs)} jobs.")
        return jobs
    except Exception as e:
        print(f"❌  Error during job scraping: {str(e)}")
        return pd.DataFrame()


def evaluate_job(title: str, description: str) -> dict:
    """Using Langchain to evaluate a job posting against the resume."""
    if not description or len(str(description)) < 50:
        return {"score": 0, "reason": "Job description too short or missing"}

    try:
        # 调用 Chain
        result: JobEvaluation = evaluation_chain.invoke({
            "resume": RESUME[:3000],  # save token
            "title": title,
            "description": description[:3000]
        })
        return {"score": result.score, "reason": result.reason}

    except Exception as e:
        print(f"⚠️  AI Evaluation Error for '{title}': {e}")
        return {"score": 0, "reason": "AI Error"}

def send_email(top_jobs: List[dict]):
    if not top_jobs:
        print("📭  No matching jobs to send.")
        return

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    subject = f"🚀 CareerScout: Top {len(top_jobs)} Jobs for {datetime.now().strftime('%Y-%m-%d')}"

    # HTML Email Template
    html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #2c3e50;">CareerScout Daily Report</h2>
            <p>Found <b>{len(top_jobs)}</b> high-match positions for you today:</p>
            <table style="border-collapse: collapse; width: 100%; max-width: 800px;">
                <tr style="background-color: #f8f9fa; text-align: left;">
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Score</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Title</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Company</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Why Match?</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Action</th>
                </tr>
        """

    for job in top_jobs:
        color = "#27ae60" if job['score'] >= 85 else "#d35400"
        html_body += f"""
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: bold; color: {color};">
                        {job['score']}
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{job['title']}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{job['company']}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; font-size: 14px; color: #555;">
                        {job['reason']}
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">
                        <a href="{job['job_url']}" style="background-color: #007bff; color: white; padding: 5px 10px; text-decoration: none; border-radius: 4px; font-size: 12px;">Apply</a>
                    </td>
                </tr>
            """

    html_body += """
            </table>
            <p style="margin-top: 20px; font-size: 12px; color: #888;">
                Powered by CareerScout-Agent using LangChain & Python.
            </p>
        </body>
        </html>
        """

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print(f"📧  Email sent successfully to {receiver}!")
    except Exception as e:
        print(f"❌  Email sending failed: {e}")


def main():
    # 1. Scraping
    df = get_jobs_data()
    if df.empty:
        return

    print(df)

    scored_jobs = []

    # 2. Evaluation Loop
    print(f"🧠  Analyzing {len(df)} jobs with AI...")

    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        # Simple filtering

        evaluation = evaluate_job(title, row.get('description', ''))

        if evaluation['score'] >= 60:  # 阈值过滤
            scored_jobs.append({
                "title": title,
                "company": row.get('company'),
                "job_url": row.get('job_url'),
                "score": evaluation['score'],
                "reason": evaluation['reason']
            })
        # 3. Sorting & Sending
        scored_jobs.sort(key=lambda x: x['score'], reverse=True)
        top_10 = scored_jobs[:10]

        send_email(top_10)

if __name__ == "__main__":
    main()