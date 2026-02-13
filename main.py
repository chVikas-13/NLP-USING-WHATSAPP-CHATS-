import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from datetime import datetime
import re
import google.generativeai as genai

# --- Set your Gemini API Key ---
GEMINI_API_KEY="AIzaSyC-erXuPBs7CyZ907Qch0QPFUHIGlvqvAg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash") # or use "gemini-pro"

# --- Streamlit UI ---
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="centered")
st.title("💬 WhatsApp Chat Analyzer with AI Summary")
st.markdown("Upload your chat and get analytics + an **AI-generated summary** using **Gemini**.")

uploaded_file = st.file_uploader("📄 Upload your WhatsApp chat (.txt file)", type=["txt"])

# --- Helper Functions ---
def parse_chat(file):
    messages = []
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - '

    for line in file:
        line = line.strip()
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if re.match(pattern, line):
            messages.append(line)
        elif messages:
            messages[-1] += " " + line

    data = []
    for msg in messages:
        parts = msg.split(" - ")
        date_time = parts[0]
        user_msg = parts[1].split(": ") if ": " in parts[1] else ["System", parts[1]]
        try:
            dt = datetime.strptime(date_time, "%d/%m/%y, %H:%M")
        except ValueError:
            dt = datetime.strptime(date_time, "%d/%m/%Y, %H:%M")
        data.append([dt, user_msg[0], user_msg[1]])

    return pd.DataFrame(data, columns=["Date", "User", "Message"])

def summarize_chat_with_gemini(chat_text):
    prompt = f"""
You're an AI that summarizes WhatsApp chats. Provide a concise summary of the following conversation:

{chat_text}

Focus on:
- Main topics discussed
- Tone of the conversation
- Any noticeable events or patterns

Summarize in bullet points.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Main Logic ---
if uploaded_file:
    chat_df = parse_chat(uploaded_file)
    chat_df["Date"] = pd.to_datetime(chat_df["Date"])

    min_date, max_date = chat_df["Date"].min(), chat_df["Date"].max()
    start_date = st.date_input("📆 Filter from date", min_value=min_date, max_value=max_date, value=min_date)
    chat_df = chat_df[chat_df["Date"] >= pd.to_datetime(start_date)]

    st.write("### 🔍 Filtered Chat Data")
    st.dataframe(chat_df.head(50))

    # Bar chart
    st.write("### 📊 Message Count per User")
    fig1, ax1 = plt.subplots()
    chat_df["User"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Messages")
    st.pyplot(fig1)

    # WordCloud
    st.write("### ☁️ Word Cloud")
    text_data = " ".join(chat_df["Message"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    # Sentiment
    st.write("### 😊 Sentiment Analysis")
    chat_df["Sentiment"] = chat_df["Message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    avg_sentiment = chat_df["Sentiment"].mean()
    st.write(f"**Average Sentiment Score:** `{avg_sentiment:.2f}` (Positive if >0, Negative if <0)")

    # Gemini summary
    st.write("### 🧠 AI Summary (Gemini)")
    if st.button("🧾 Generate AI Summary"):
        with st.spinner("Generating summary with Gemini..."):
            # Limit input to ~4000 characters
            summary_input = " ".join(chat_df["Message"].dropna())[:4000]
            try:
                summary = summarize_chat_with_gemini(summary_input)
                st.success("✅ Summary Generated!")
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Error during summarization: {str(e)}")
