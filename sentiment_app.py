import streamlit as st
import pandas as pd
import json
import re
from openai import OpenAI

# ─── Configuration ───
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4"  # or gpt-4-turbo, gpt-3.5-turbo

st.title("Community Clinic Feedback Sentiment Analysis")

# ─── File uploader ───
uploaded = st.file_uploader("Upload comments JSON or CSV", type=["json", "csv"])

if uploaded:
    # Load comments into DataFrame
    if uploaded.name.endswith('.json'):
        comments = json.load(uploaded)
        df = pd.DataFrame({"comment": comments})
    else:
        df = pd.read_csv(uploaded)
        if 'comment' not in df.columns:
            st.error("CSV must contain a 'comment' column.")
            st.stop()

    st.subheader("Loaded Comments")
    st.dataframe(df)

    if st.button("Analyze Sentiment"):
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
        df['sentiment'] = analyze_sentiments(df['comment'].tolist())

        st.subheader("Sentiment Results")
        st.dataframe(df)

        # Display distribution
        st.subheader("Sentiment Distribution")
        dist = df['sentiment'].value_counts().reset_index()
        dist.columns = ['Sentiment', 'Count']
        st.bar_chart(data=dist.set_index('Sentiment'))

        # Comments with assessed labels
        st.subheader("Comments with Assessed Sentiment")
        for idx, row in df.iterrows():
            st.markdown(f"**Comment:** {row['comment']}")
            st.markdown(f"**Assessed Sentiment:** {row['sentiment']}  ")
            st.markdown('---')
