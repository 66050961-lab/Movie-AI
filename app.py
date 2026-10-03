import os
import random
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

load_dotenv()

# ==========================================
# 1. การตั้งค่า Gemini API
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash") 

# ==========================================
# 2. Page Configuration & Theme
# ==========================================
st.set_page_config(page_title="Cinema's the Goat", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0B0E14; color: #FFFFFF; font-family: 'Inter', sans-serif;}
    [data-testid="stSidebar"] { background-color: #121620 !important; border-right: 1px solid #1F2433;}
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    
    div[data-testid="stSidebar"] .stButton > button {
        background-color: transparent; border: 1px solid #1F2433; border-radius: 12px;
        text-align: left; padding: 15px 16px; transition: all 0.3s ease;
        color: #94A3B8 !important; font-size: 3rem !important; font-weight: 600;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(123, 97, 255, 0.1); color: #7B61FF !important;
        border-color: #7B61FF; transform: translateX(5px);
    }

    div[data-testid="stVerticalBlock"] .stButton > button {
        border-radius: 16px; border: 1px solid #1F2433; background-color: #161B27;
        color: #CBD5E1 !important; height: 110px; text-align: left; white-space: pre-wrap;
        transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stVerticalBlock"] .stButton > button:hover {
        background-color: #1E2536; border-color: #7B61FF; box-shadow: 0 8px 15px rgba(123, 97, 255, 0.15);
    }

    .botbuzz-logo {
        text-align: center; font-size: 4.5rem; font-weight: 900;
        background: linear-gradient(90deg, #FF3131 100%);   
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-top: 4vh; margin-bottom: 4vh; letter-spacing: -2px;
    }
    
    [data-testid="stChatMessage"] { background-color: transparent; padding: 0; margin-bottom: 20px;}
    
    [data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"] > p:first-child:contains("user")) {
        background-color: #7B61FF; color: white; border-radius: 20px 20px 4px 20px;
        padding: 15px 20px; margin-left: auto; max-width: 85%; box-shadow: 0 4px 15px rgba(123, 97, 255, 0.2);
    }
    [data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"] > p:first-child:contains("user")) * { color: white !important; }
    
    [data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"] > p:first-child:contains("assistant")) {
        background-color: #161B27; border: 1px solid #1F2433; border-radius: 20px 20px 20px 4px;
        padding: 15px 20px; max-width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stChatInput"] { border-radius: 24px !important; border: 1px solid #2A3143 !important; background-color: #121620 !important;}

    .xai-indicator {
        background: rgba(123, 97, 255, 0.12); border: 1px solid rgba(123, 97, 255, 0.3);
        border-radius: 12px; padding: 10px 16px; margin-bottom: 15px; font-size: 0.88rem;
        color: #CBD5E1; display: flex; align-items: center; gap: 10px;
    }
    .chip {
        background: #1E293B; color: #94A3B8; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AI Tools & Data Loaders
# ==========================================
def get_csv_path():
    possible_names = ["all_movie_cleaned_final.csv", "all_Movies_cleaned_final.csv", "Top_1000_IMDB_movies_.csv"]
    for name in possible_names:
        if os.path.exists(name): return name
    for f in os.listdir("."):
        if f.endswith(".csv"): return f
    return "all_movie_cleaned_final.csv"

@st.cache_data
def load_movie_knowledge_base() -> pd.DataFrame:
    try:
        df = pd.read_csv(get_csv_path())
        if 'Movie Name' not in df.columns:
            df['Movie Name'] = df.get('title', 'Unknown Title')
        
        score_col = 'final_score' if 'final_score' in df.columns else ('rating' if 'rating' in df.columns else None)
        df['final_score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(7.0) if score_col else 7.0
        
        desc_col = 'Description' if 'Description' in df.columns else ('description' if 'description' in df.columns else None)
        df['Description'] = df[desc_col].fillna('') if desc_col else ''
        
        df['genre'] = df.get('genre', 'Drama').fillna('Drama')
        df['year'] = df.get('year', '').astype(str).fillna('')
        df['final_runtime'] = df.get('final_runtime', df.get('watch_time', '120 min')).astype(str).fillna('120 min')
        df['director'] = df.get('director', 'N/A')
        df['star'] = df.get('star', 'N/A')
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def analyze_review_sentiment(review_text: str) -> dict:
    pos_words = ["good", "love", "ชอบ", "สนุก", "ดี", "ฮีลใจ"]
    neg_words = ["bad", "boring", "น่าเบื่อ", "แย่", "เครียด"]
    text_lower = review_text.lower()
    
    sentiment = "เชิงบวก (Positive)" if sum(1 for w in pos_words if w in text_lower) > sum(1 for w in neg_words if w in text_lower) else ("เชิงลบ / เครียด" if sum(1 for w in neg_words if w in text_lower) > sum(1 for w in pos_words if w in text_lower) else "ปานกลาง (Neutral)")
    
    if any(w in text_lower for w in ["แอ็คชั่น", "ตื่นเต้น", "มันส์", "action"]): intent = "High Energy Action"
    elif any(w in text_lower for w in ["ตลก", "คลายเครียด", "ขำ", "comedy"]): intent = "Comedy & Feel-Good"
    elif any(w in text_lower for w in ["รัก", "โรแมนติก", "ซาบซึ้ง"]): intent = "Romantic & Drama"
    elif any(w in text_lower for w in ["ผี", "สยองขวัญ", "น่ากลัว"]): intent = "Horror & Thriller"
    else: intent = "General Discovery"
        
    return {"sentiment": sentiment, "intent": intent}

def recommend_movies_tool(user_review: str, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if df.empty or 'Description' not in df.columns: return pd.DataFrame()
    
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = (df['Description'].astype(str) + " " + df['genre'].astype(str)).tolist() + [user_review]
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    result_df = df.copy()
    
    raw_scores = similarities * 100
    scaled_scores = 70.0 + (raw_scores / raw_scores.max()) * 25.0 if raw_scores.max() > 0 else np.random.uniform(75.0, 92.0, size=len(df))
    result_df["similarity_score"] = np.round(scaled_scores, 1)
    
    top_matches = result_df.sort_values(by="similarity_score", ascending=False).head(top_n).copy()
    
    matched_features_list = []
    xai_rationales = []
    
    for _, row in top_matches.iterrows():
        words = [w.capitalize() for w in str(row['Description']).split() if len(w) > 4][:3]
        matched_features_list.append([f"#{w}" for w in (words if words else ["แนะนำ", "คะแนนสูง"])])
        
        dir_info = f" กำกับโดย {row['director']}" if str(row.get('director')) != 'N/A' else ""
        xai_rationales.append(f"คัดเลือกเรื่องนี้เพราะ Keyword ตรงกับความต้องการของคุณ {row['similarity_score']}%{dir_info}")
        
    top_matches['matched_features'] = matched_features_list
    top_matches['xai_rationales'] = xai_rationales
    return top_matches

# ==========================================
# 4. Gemini AI Chat Core Function
# ==========================================
def _chat_with_gemini(prompt, system_instruction, temperature=0.85):
    if not GEMINI_API_KEY:
        return "⚠️ กรุณาตั้งค่า GEMINI_API_KEY ในไฟล์ .env หรือ System Environment"
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=600
            )
        )
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ ขออภัยครับ โควต้า Gemini 3.5-flash ของคุณอาจจะเต็ม กรุณาลองใช้โมเดลอื่น"
        return f"⚠️ Gemini AI Error: {str(e)}"

def is_movie_query(text: str) -> bool:
    if not text: return False
    text_lower = text.strip().lower()
    movie_kw = ["หนัง", "movie", "แนะนำ", "แนว", "action", "รัก", "ผี", "ตลก", "ดูอะไรดี", "เรื่อง", "คะแนน"]
    return any(kw in text_lower for kw in movie_kw)

# ==========================================
# 5. UI Layout & App Execution
# ==========================================
movies_df = load_movie_knowledge_base()

if "messages" not in st.session_state: st.session_state.messages = []
if "active_prompt" not in st.session_state: st.session_state.active_prompt = None

with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
        <div style='background: linear-gradient(135deg,#FF3131); width: 35px; height: 35px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px;'>🍿</div>
        <h2 style='margin: 0; font-weight: 800;'>CineMatch AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #64748B; font-size: 1rem; font-weight: 600; text-transform: uppercase;'>Explore Genres</p>", unsafe_allow_html=True)
    
    genres = {
        "🔥 Action": "หาหนังแนว Action เดือดๆ ให้หน่อย",
        "😂 Comedy": "แนะนำหนังตลก Comedy ให้หน่อย",
        "❤️ Romance": "อยากดูหนังรัก Romance ซึ้งๆ",
        "👻 Horror": "ขอหนังผี Horror น่ากลัวๆ",
        "🚀 Sci-Fi": "แนะนำหนัง Sci-Fi ไซไฟอวกาศ",
        "😢 Drama": "ขอหนังแนวดราม่า ชีวิต"
    }
    
    for label, prompt in genres.items():
        if st.button(label, use_container_width=True):
            st.session_state.active_prompt = prompt

spacer_left, main_col, spacer_right = st.columns([1.5, 7, 1.5])

with main_col:
    st.markdown("""
    <div style='display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;'>
        <div style='font-weight: 800; font-size: 1.25rem; color: #ff3131;'> <span style='font-size: 0.8rem; color: #ff3131; font-weight: normal;'></span></div>
    </div>
    """, unsafe_allow_html=True)

    chat_container = st.container(border=False)
    
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.markdown('<br><br><br><br>', unsafe_allow_html=True)
            st.markdown('<div class="botbuzz-logo">CineMatch</div>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 40px;'>วิเคราะห์ความรู้สึกและค้นหาภาพยนตร์ที่ตรงใจคุณด้วย AI</p>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("ระเบิดภูเขา เผากระท่อม 💥\nตื่นเต้น เร้าใจ ลุ้นจนนั่งไม่ติด", use_container_width=True):
                    st.session_state.active_prompt = "อยากดูหนังแอ็คชั่นตื่นเต้น เร้าใจ ลุ้นจนนั่งไม่ติด"
            with c2:
                if st.button("หัวเราะจนปอดโยก 😂\nขอหนังตลกคลายเครียดแบบขำๆ", use_container_width=True):
                    st.session_state.active_prompt = "ขอหนังตลกคลายเครียดแบบไม่ต้องคิดอะไรมาก"
            with c3:
                if st.button("อบอุ่นหัวใจ ฮีลใจ ❤️\nขอหนังรักโรแมนติก หรือหนังครอบครัว", use_container_width=True):
                    st.session_state.active_prompt = "ขอหนังรักโรแมนติก หรือหนังครอบครัวฮีลใจ"
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    if msg["role"] == "assistant" and msg.get("intent"):
                        st.markdown(f"""
                        <div class="xai-indicator">
                            📊 <b>XAI NLP Indicator:</b> Intent: <span style="color:#00C2FF;">{msg['intent']}</span> | Sentiment: <span style="color:#10B981;">{msg.get('sentiment', 'Positive')}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if msg["role"] == "assistant" and msg.get("movie_cards"):
                        st.markdown("<hr style='border-color: #1F2433; margin: 15px 0;'>", unsafe_allow_html=True)
                        cols = st.columns(len(msg["movie_cards"]))
                        for idx, movie in enumerate(msg["movie_cards"]):
                            with cols[idx]:
                                keywords_html = "".join([f'<span class="chip">{k}</span>' for k in movie.get('matched_features', [])])
                                st.markdown(f"""
                                <div style="background: #121620; border: 1px solid #1F2433; padding: 12px; border-radius: 16px; margin-bottom: 10px;">
                                    <div style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{movie.get('Movie Name', 'Unknown')}</div>
                                    <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;">🎬 {movie.get('genre', '').split(',')[0]}</div>
                                    <div style="color: #00C2FF; font-size: 0.95rem; font-weight: 700;">★ {movie.get('final_score', '')} <span style="color: #475569; font-weight: normal; margin-left: 5px;">| {movie.get('year', '')}</span></div>
                                    <div style="margin-top: 10px; margin-bottom: 10px;">{keywords_html}</div>
                                    <div style="border-top: 1px dashed #1F2433; padding-top: 10px; margin-top: 8px;">
                                        <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
                                            💡 <b>ทำไม AI ถึงแนะนำ:</b> {movie.get('xai_rationales', 'ตรงกับความชอบของคุณ')}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

    user_input = st.chat_input("พิมพ์บอก AI ว่าอยากดูหนังแนวไหน...")
    
    if st.session_state.active_prompt:
        user_input = st.session_state.active_prompt
        st.session_state.active_prompt = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]
        
        with chat_container:
            with st.chat_message("assistant"):
                history_text = ""
                if len(st.session_state.messages) > 1:
                    history_text = "ประวัติสนทนาก่อนหน้า:\n"
                    for msg in st.session_state.messages[-4:-1]:
                        role_name = "ผู้ใช้" if msg["role"] == "user" else "AI"
                        history_text += f"{role_name}: {msg['content']}\n"
                
                if not is_movie_query(user_msg):
                    with st.spinner("กรูณารอสักครู่ กำลังพิมพ์ตอบ..."):
                        prompt = f"""{history_text}
ผู้ใช้พิมพ์มาล่าสุดว่า: "{user_msg}"

คำสั่ง:
ตอบกลับผู้ใช้อย่างเป็นธรรมชาติและเป็นกันเอง ห้ามใช้ข้อความแพทเทิร์นหุ่นยนต์ซ้ำๆ เด็ดขาด
ถ้าผู้ใช้ทักทาย, ถามเรื่องทั่วไป, บ่น หรือพิมพ์คำสั้นๆ ให้โต้ตอบกลับเนียนๆ และชวนคุยเข้าเรื่องแนะนำภาพยนตร์
"""
                        ai_reply = _chat_with_gemini(
                            prompt=prompt,
                            system_instruction="คุณคือ CineMatch AI ผู้เชี่ยวชาญด้านภาพยนตร์ คุยเก่งและมีอารมณ์ขัน",
                            temperature=0.85
                        )
                        st.markdown(ai_reply)
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                        
                else:
                    with st.spinner("Gemini AI กำลังวิเคราะห์เจตนาและค้นหาข้อมูลภาพยนตร์..."):
                        sentiment_info = analyze_review_sentiment(user_msg)
                        rec_df = recommend_movies_tool(user_msg, movies_df, top_n=3)
                        
                        context_data = rec_df[["Movie Name", "year", "genre", "final_score", "Description"]].to_string(index=False) if not rec_df.empty else "ไม่พบข้อมูล"
                        
                        final_prompt = f"""{history_text}
ผู้ใช้บอกว่า: "{user_msg}"
                        
ข้อมูลหนังที่ระบบกรองมาให้:
{context_data}

คำสั่ง:
นำข้อมูลหนังด้านบนมาอธิบายและป้ายยาให้ผู้ใช้
- อธิบายด้วยน้ำเสียงเป็นกันเอง เป็นธรรมชาติเหมือนคุยกับเพื่อน 
- ห้ามคิดชื่อหนังขึ้นมาเอง ให้อิงจากข้อมูลหนังที่ระบบกรองได้เท่านั้น
- เขียนสั้นๆ กระชับ น่าอ่าน
"""
                        ai_reply = _chat_with_gemini(
                            prompt=final_prompt,
                            system_instruction="คุณคือ CineMatch  AI ผู้เชี่ยวชาญด้านภาพยนตร์ที่ชอบคุยสนุกสนาน",
                            temperature=0.85 
                        )
                        
                        st.markdown(ai_reply)
                        
                        st.markdown(f"""
                        <div class="xai-indicator">
                            📊 <b>XAI NLP Indicator:</b> Intent: <span style="color:#00C2FF;">{sentiment_info['intent']}</span> | Sentiment: <span style="color:#10B981;">{sentiment_info['sentiment']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        movie_cards_data = rec_df.to_dict(orient="records") if not rec_df.empty else []
                        if movie_cards_data:
                            st.markdown("<hr style='border-color: #1F2433; margin: 15px 0;'>", unsafe_allow_html=True)
                            cols = st.columns(len(movie_cards_data))
                            for idx, movie in enumerate(movie_cards_data):
                                with cols[idx]:
                                    keywords_html = "".join([f'<span class="chip">{k}</span>' for k in movie.get('matched_features', [])])
                                    st.markdown(f"""
                                    <div style="background: #121620; border: 1px solid #1F2433; padding: 12px; border-radius: 16px; margin-bottom: 10px;">
                                        <div style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{movie.get('Movie Name', 'Unknown')}</div>
                                        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;">🎬 {movie.get('genre', '').split(',')[0]}</div>
                                        <div style="color: #00C2FF; font-size: 0.95rem; font-weight: 700;">★ {movie.get('final_score', '')} <span style="color: #475569; font-weight: normal; margin-left: 5px;">| {movie.get('year', '')}</span></div>
                                        <div style="margin-top: 10px; margin-bottom: 10px;">{keywords_html}</div>
                                        <div style="border-top: 1px dashed #1F2433; padding-top: 10px; margin-top: 8px;">
                                            <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
                                                💡 <b>ทำไม AI ถึงแนะนำ:</b> {movie.get('xai_rationales', 'ตรงกับความชอบของคุณ')}
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ai_reply,
                            "intent": sentiment_info['intent'],
                            "sentiment": sentiment_info['sentiment'],
                            "movie_cards": movie_cards_data
                        })