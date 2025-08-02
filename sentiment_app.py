import streamlit as st
import pandas as pd
import json
import re
from openai import OpenAI
import plotly.express as px

# ─── Configuration ───
API_KEY = st.secrets.get("OPENAI_API_KEY")
client  = OpenAI(api_key=API_KEY)
MODEL    = "gpt-4"  # or gpt-4-turbo, gpt-3.5-turbo

st.title("Community Clinic Feedback Sentiment Analysis")

# ─── Input: File uploader and manual text entry ───
st.sidebar.header("Input Options")
uploaded = st.sidebar.file_uploader("Upload comments JSON/CSV", type=["json", "csv"])
manual_text = st.sidebar.text_area(
    "Or paste comments (one per line)", height=200
)

# Build comments list
comments = []
if uploaded:
    if uploaded.name.endswith('.json'):
        comments = json.load(uploaded)
    else:
        df_temp = pd.read_csv(uploaded)
        if 'comment' in df_temp.columns:
            comments = df_temp['comment'].dropna().astype(str).tolist()
        else:
            st.sidebar.error("CSV must contain a 'comment' column.")
elif manual_text:
    comments = [line.strip() for line in manual_text.splitlines() if line.strip()]

if not comments:
    st.info("Please upload a file or enter comments in the text box to analyze.")
    st.stop()

# ─── Sentiment analysis ───
@st.cache_data(show_spinner=False)
def analyze_sentiments(comments_list):
    results = []
    for text in comments_list:
        messages = [
            {"role": "system", "content": (
                "You are a sentiment analysis assistant. "
                "Label each comment as Positive, Neutral, or Negative."
            )},
            {"role": "assistant", "content": "Example: 'The clinic meets my expectations.' → Neutral"},
            {"role": "assistant", "content": "Example: 'I appreciated how kind the nurses were.' → Positive"},
            {"role": "assistant", "content": "Example: 'I waited over an hour past my appointment.' → Negative"},
            {"role": "user", "content": f"Comment: \"{text}\""}
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
        sentiment = resp.choices[0].message.content.strip()
        results.append(sentiment)
    return results

# Run analysis
sentiments = analyze_sentiments(comments)
df = pd.DataFrame({"comment": comments, "sentiment": sentiments})

# ─── Theme extraction (top 5) ───
@st.cache_data(show_spinner=False)
def extract_themes(comments_list, n=5):
    text_blob = "\n".join(comments_list)
    prompt = (
        f"Extract exactly {n} main themes from this clinic feedback. "
        "Output as a JSON array of strings, no extra text.\n\n" + text_blob
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a themes extraction assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3
    )
    raw = resp.choices[0].message.content
    # strip fences, fix commas
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r",\s*]", "]", clean)
    m = re.search(r"(\[.*\])", clean, flags=re.DOTALL)
    return json.loads(m.group(1))

themes = extract_themes(comments, n=5)
st.subheader("Top 5 Themes")
for i, theme in enumerate(themes, 1):
    st.write(f"{i}. {theme}")

# ─── Visualization ───
st.subheader("Sentiment Distribution")
counts = df['sentiment'].value_counts().reset_index()
counts.columns = ['Sentiment', 'Count']
fig = px.pie(counts, names='Sentiment', values='Count', title='Feedback Sentiment')
st.plotly_chart(fig, use_container_width=True)

# ─── Download results ───
st.subheader("Download Analysis")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download comments with sentiment",
    data=csv,
    file_name="comments_sentiment.csv",
    mime='text/csv'
)

# ─── View raw input in separate window ───
st.subheader("Raw Comments")
st.markdown("<a target=\"_blank\" href=\"#\">Open Comments in New Tab</a>", unsafe_allow_html=True)
st.write(df['comment'])
