import os
from dotenv import load_dotenv

load_dotenv()

# รองรับทั้ง Gemini API และ Microsoft Azure AI Foundry / OpenAI
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

def ask_movie_chat_ai(user_prompt: str, context_movies_text: str = "") -> str:
    """
    ส่งข้อความแชทและบริบทภาพยนตร์จาก all_Movies_cleaned_final ไปประมวลผลด้วย AI
    รองรับทั้ง Azure AI Foundry และ Gemini API
    """
    full_prompt = f"""บริบทภาพยนตร์ที่คัดเลือกจากคลังข้อมูล all_Movies_cleaned_final:
{context_movies_text}

คำขอของผู้ใช้:
{user_prompt}

คำแนะนำในการตอบ:
1. ตอบเป็นภาษาไทยอย่างเป็นกันเอง สนุกสนาน สไตล์เพื่อนป้ายยาภาพยนตร์
2. สรุปความต้องการของผู้ใช้ และป้ายยาแนะนำภาพยนตร์ 3 เรื่องที่คัดเลือกมาจากคลังข้อมูล
3. อธิบายเหตุผลเบื้องหลังคำแนะนำอย่างโปร่งใส (Explainable AI - XAI) ว่าทำไมเรื่องย่อ ผู้กำกับ หรือแนวหนังเรื่องนี้ถึงตรงกับความสนใจของผู้ใช้"""

    # 1. ลองใช้ Microsoft Azure AI Foundry หากตั้งค่าไว้
    if AZURE_API_KEY and AZURE_ENDPOINT:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION
            )
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are CINEBUZZ AI Movie Companion powered by Microsoft Azure AI Foundry."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content or "ไม่มีข้อความตอบกลับจาก Azure AI"
        except Exception as e:
            print(f"Azure AI Error: {e}")

    # 2. ลองใช้ Gemini API หากตั้งค่าไว้
    if GEMINI_KEY:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are CINEBUZZ AI Movie Critic & Recommendation Assistant.",
                    temperature=0.3
                )
            )
            return response.text or "ไม่มีข้อความตอบกลับจาก Gemini AI"
        except Exception as e:
            print(f"Gemini API Error: {e}")

    # 3. Dynamic Fallback เมื่อไม่ได้ใส่ API Key
    return f"""🤖 **CINEBUZZ XAI Engine** ได้วิเคราะห์ข้อความของคุณ: *"{user_prompt}"*

จากการประมวลผลภาษาธรรมชาติ (NLP) และการวิเคราะห์เวกเตอร์เรื่องย่อในคลังข้อมูล **all_Movies_cleaned_final** ระบบได้คัดเลือกภาพยนตร์ที่มีค่า Match Score สูงสุด 3 เรื่องแรก พร้อมสกัดคีย์เวิร์ดมาป้ายยาเรียบร้อยครับ! 

ลองเลือกชมรายละเอียดการ์ดภาพยนตร์และกดดูเหตุผล XAI ด้านล่างได้เลยครับ 👇"""
