import os
import json
import hashlib
import re
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from google.cloud import texttospeech
from google.oauth2 import service_account
import asyncio
from io import BytesIO

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def get_google_tts_client():
    """Initialize Google TTS client with credentials from environment variable"""
    if GOOGLE_CREDENTIALS_JSON:
        credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return texttospeech.TextToSpeechClient(credentials=credentials)
    else:
        # Fallback to default credentials if available
        return texttospeech.TextToSpeechClient()

def split_text_into_sentences(text, max_length=150):
    """Split text into smaller sentences for Chirp3"""
    sentences = re.split(r'([。！？；])', text)
    
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    
    final_result = []
    for sentence in result:
        if len(sentence) > max_length:
            parts = re.split(r'([，、])', sentence)
            temp = ""
            for part in parts:
                if len(temp + part) > max_length and temp:
                    final_result.append(temp)
                    temp = part
                else:
                    temp += part
            if temp:
                final_result.append(temp)
        else:
            final_result.append(sentence)
    
    return [s.strip() for s in final_result if s.strip()]

def generate_tts_chirp(text):
    """Generate Chinese TTS audio using Google Cloud Chirp3"""
    try:
        client = get_google_tts_client()
        
        sentences = split_text_into_sentences(text, max_length=150)
        
        all_audio = b""
        for sentence in sentences:
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="cmn-CN",
                name="cmn-CN-Chirp3-HD-Aoede",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            all_audio += response.audio_content
        
        return all_audio
    
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None

def generate_content_with_deepseek(topic):
    """Generate all content using DeepSeek API"""
    prompt = f"""You are a Chinese language teaching assistant. Create learning materials about the topic: "{topic}"

Please generate a JSON response with the following structure:
{{
  "main_text": "A text in Simplified Chinese at HSK4 level about {topic}. Should be 150-200 characters long, natural and engaging.",
  "vocabulary": [
    {{"english": "English translation", "chinese": "Chinese word/phrase from the text", "pinyin": "pinyin with tone marks"}},
    // 10-15 items total - must be HSK4 or HSK5 level words, phrases, or expressions taken directly from the main_text
  ],
  "opinion_texts": {{
    "positive": "A natural Chinese text (HSK4 level, 100-150 characters) giving a positive perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context.",
    "negative": "A natural Chinese text (HSK4 level, 100-150 characters) giving a critical/negative perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context.",
    "balanced": "A natural Chinese text (HSK4 level, 100-150 characters) giving a balanced perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context."
  }},
  "discussion_questions": [
    "Question 1 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 2 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 3 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 4 in Chinese (HSK4 level) - should prompt discussion, not just comprehension"
  ]
}}

Important requirements:
1. All vocabulary items MUST come from the main_text
2. Vocabulary should be HSK4-5 level (not easier levels)
3. Opinion texts should sound natural - vocabulary can be adjusted to fit context
4. Discussion questions should encourage personal opinions and deeper thinking
5. Return ONLY valid JSON, no additional text"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Chinese language teaching expert who creates engaging, natural content at HSK4 level. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        return json.loads(content)
    
    except Exception as e:
        print(f"DeepSeek API Error: {str(e)}")
        return None

def create_vocabulary_file(vocabulary, topic):
    """Create tab-delimited vocabulary file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{topic}_{timestamp}_vocabulary.txt"
    
    content = ""
    for item in vocabulary:
        content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
    
    return filename, content

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "欢迎! Welcome to the Chinese Learning Bot!\n\n"
        "Simply send me a topic (e.g., '大城市生活的压力', 'environmental protection', etc.) "
        "and I'll create:\n"
        "📄 A reading text\n"
        "📝 Vocabulary list\n"
        "🎤 3 opinion texts with audio\n"
        "💬 Discussion questions\n\n"
        "Just type your topic to begin!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "How to use:\n"
        "1. Send me any topic in English or Chinese\n"
        "2. Wait while I generate materials (takes about 30-60 seconds)\n"
        "3. You'll receive:\n"
        "   - Vocabulary file (.txt)\n"
        "   - 3 audio files with different perspectives\n"
        "   - 3 opinion texts\n"
        "   - Discussion questions\n\n"
        "Example topics:\n"
        "- 社交媒体的影响\n"
        "- work-life balance\n"
        "- 环境保护\n"
        "- modern technology"
    )

async def handle_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle topic message and generate all materials"""
    topic = update.message.text.strip()
    
    await update.message.reply_text(f"📚 正在创建关于 '{topic}' 的学习材料...\n"
                                   f"Creating materials about '{topic}'...\n"
                                   f"This will take about 30-60 seconds...")
    
    # Generate content with DeepSeek
    content = generate_content_with_deepseek(topic)
    
    if not content:
        await update.message.reply_text("❌ Sorry, there was an error generating content. Please try again.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_')[:30]
    
    try:
        # Send main text
        await update.message.reply_text(f"📖 **Main Text:**\n\n{content['main_text']}", parse_mode='Markdown')
        
        # Validate vocabulary items are from main text
        main_text = content['main_text']
        missing_vocab = []
        for item in content['vocabulary']:
            if item['chinese'] not in main_text:
                missing_vocab.append(item['chinese'])
        
        if missing_vocab:
            await update.message.reply_text(
                f"⚠️ Note: {len(missing_vocab)} vocabulary items may not be from the main text. "
                f"This sometimes happens with AI generation."
            )
        
        # Create and send vocabulary file
        vocab_filename, vocab_content = create_vocabulary_file(content['vocabulary'], safe_topic)
        vocab_file = BytesIO(vocab_content.encode('utf-8'))
        vocab_file.name = vocab_filename
        await update.message.reply_document(document=vocab_file, filename=vocab_filename)
        
        # Generate and send opinion texts with audio
        perspectives = [
            ("positive", "Positive View", "😊"),
            ("negative", "Critical View", "🤔"),
            ("balanced", "Balanced View", "⚖️")
        ]
        
        for key, name, emoji in perspectives:
            opinion_text = content['opinion_texts'][key]
            
            # Send text
            await update.message.reply_text(f"{emoji} **{name}:**\n\n{opinion_text}", parse_mode='Markdown')
            
            # Generate and send audio
            await update.message.reply_text(f"🎤 Generating audio for {name.lower()}...")
            audio_data = generate_tts_chirp(opinion_text)
            
            if audio_data:
                audio_filename = f"{safe_topic}_{timestamp}_{key}.mp3"
                audio_file = BytesIO(audio_data)
                audio_file.name = audio_filename
                await update.message.reply_audio(audio=audio_file, filename=audio_filename)
            else:
                await update.message.reply_text(f"❌ Audio generation failed for {name}. Text is still available above.")
        
        # Send discussion questions
        questions_text = "💬 **Discussion Questions:**\n\n"
        for i, question in enumerate(content['discussion_questions'], 1):
            questions_text += f"{i}. {question}\n"
        
        await update.message.reply_text(questions_text, parse_mode='Markdown')
        
        await update.message.reply_text("✅ All materials created! Happy learning! 加油！")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error sending materials: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    """Start the bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topic))
    
    # Run the bot
    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
