import os
import json
import requests
import re
from typing import List, Optional, Dict, Any
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, abort
from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from notion_client import Client
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'replace-this-with-secure-random-in-prod')

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # <-- Gemini Key
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

# --- Initialize Gemini (defensive) ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Some SDK usage: keep a placeholder name; we'll call via genai or model wrapper later
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        print("Warning: couldn't configure Gemini client:", e)
        gemini_model = None
else:
    gemini_model = None
    print("Warning: GOOGLE_API_KEY not set; Gemini calls will be skipped.")

# --- Initialize Notion (defensive) ---
notion = None
if NOTION_API_KEY:
    try:
        notion = Client(auth=NOTION_API_KEY)
    except Exception as e:
        print("Warning: couldn't initialize Notion client:", e)
else:
    print("Warning: NOTION_API_KEY not set; Notion integration disabled.")

# --- CACHE ---
CACHE_FILE = "symptom_cache.json"
def load_cache() -> Dict[str, str]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache_data: Dict[str, str]):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)

symptom_cache = load_cache()

# --- Helper: safe Gemini classify with multiple fallbacks ---
def call_gemini_for_category(symptom_text: str) -> str:
    """Return category name or 'General' on failure. Try multiple response shapes."""
    if not gemini_model:
        return "General"

    prompt = """
        You are a highly trained AI medical triage assistant. Your only job is to classify symptoms into ONE of the following three medical categories:

        1. General:
        - Non-urgent conditions such as: fever, cold, cough, sore throat, minor injuries, headaches, stomach pain, body aches, skin rashes, fatigue, nausea, diarrhea, constipation, menstrual cramps, acidity, back pain, etc.

        2. Emergency:
        - Life-threatening or urgent conditions such as: chest pain, difficulty breathing, seizures, stroke symptoms (face drooping, arm weakness, speech difficulty), unconsciousness, high fever in infants, severe allergic reactions, heavy bleeding, fractures, serious burns, poisoning, head trauma, vision loss, etc.

        3. Mental Health:
        - Psychological or behavioral symptoms such as: sadness, depression, suicidal thoughts, self-harm, anxiety, panic attacks, mood swings, hallucinations, delusions, anger issues, insomnia, social withdrawal, emotional numbness, PTSD, eating disorders, substance abuse, etc.

        Instructions:
        - Return only ONE of these: "General", "Emergency", or "Mental Health".
        - Be very strict. Do NOT default to "General" if the symptoms match Emergency or Mental Health.
        - NO extra words. Only return the category in **one word**.

        Now classify the following symptom:
        Symptom: {user_input}
        Category:
        """

    try:
        # Preferred: try the model wrapper if available
        if hasattr(gemini_model, "generate_content"):
            response = gemini_model.generate_content(prompt)
            # try common attribute patterns
            if hasattr(response, "text") and isinstance(response.text, str):
                category = response.text.strip()
                return category
            # some SDKs use candidates -> content -> parts -> text
            if getattr(response, "candidates", None):
                cand = response.candidates[0]
                # try a few nested shapes
                for attr in ("content", "message", "output"):
                    part = getattr(cand, attr, None)
                    if part:
                        # If it's a dict-like
                        if isinstance(part, dict):
                            # try retrieving textual parts
                            # many shapes: {'parts': [{'type':'output_text','text': '...'}]}
                            parts = part.get("parts") or part.get("text")
                            if isinstance(parts, list) and parts:
                                first = parts[0]
                                if isinstance(first, dict) and "text" in first:
                                    return first["text"].strip()
                        # fallback:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            return text.strip()
            # final fallback: try str(response)
            return str(response).strip().splitlines()[0]
        # Older SDK: genai.generate
        elif hasattr(genai, "generate"):
            res = genai.generate(model="gemini-1.5-pro", prompt=prompt)
            # res.output may contain text
            text = getattr(res, "text", None) or res.output
            if isinstance(text, str):
                return text.strip().splitlines()[0]
    except Exception as e:
        print("Gemini call failed:", e)

    return "General"

# --- DOCTOR TYPE NODES ---
def pediatrics_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['doctor_type'] = "Pediatrician"
    return s

def geriatrics_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['doctor_type'] = "Geriatrician"
    return s

def general_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['doctor_type'] = "General Practitioner"
    return s

def mentalhealth_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['doctor_type'] = "Mental Health Specialist"
    return s

def emergency_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['doctor_type'] = "Emergency Doctor"
    return s

def show_time_slots_node(s: Dict[str, Any]) -> Dict[str, Any]:
    s['time_slots'] = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM"]
    return s

def confirm_appointment_node(s: Dict[str, Any]) -> Dict[str, Any]:
    if not s.get('chosen_slot'):
        s['final_message'] = "Appointment booking cancelled."
    else:
        s['final_message'] = (
            f"Dear {s.get('name')}, your appointment with a {s.get('doctor_type')} "
            f"for '{s.get('symptom')}' is confirmed for {s['chosen_slot']}."
        )
    s['current_step'] = 'confirmed'
    return s

# --- ROUTERS ---
def age_router(s: Dict[str, Any]) -> str:
    age = s.get('age')
    try:
        age_val = int(age)
    except Exception:
        return "get_symptom"  # default to symptom step if invalid
    if age_val < 18:
        return "pediatrics"
    elif age_val > 65:
        return "geriatrics"
    else:
        return "get_symptom"

def symptom_router(s: Dict[str, Any]) -> str:
    cat = str(s.get('category', '')).lower()
    if "general" in cat:
        return "general"
    elif "emergency" in cat:
        return "emergency"
    elif "mental" in cat:
        return "mentalhealth"
    else:
        return "general"

# --- GEMINI Symptom Classification (improved) ---
def classify_symptom_node(state: Dict[str, Any]) -> Dict[str, Any]:
    symptom_text = str(state.get('symptom', '')).strip().lower()
    if not symptom_text:
        state['category'] = "General"
        return state

    found_category = None

    for cached_symptom, cached_category in symptom_cache.items():
        cs_lower = cached_symptom.lower()

        # Method 1: Direct substring in either direction
        if cs_lower in symptom_text or symptom_text in cs_lower:
            found_category = cached_category
            break

        # Method 2: Word overlap check (≥50% of cached words match input)


    if found_category:
        category = found_category
    else:
        # Fallback to Gemini with robust wrapper
        category = call_gemini_for_category(symptom_text)
        if category not in ["General", "Emergency", "Mental Health"]:
            # Do some normalization in case gemini returns like "Emergency."
            category_norm = category.strip().title()
            if category_norm in ["General", "Emergency", "Mental Health"]:
                category = category_norm
            else:
                category = "General"

        # Save to cache
        symptom_cache[symptom_text] = category
        try:
            save_cache(symptom_cache)
        except Exception as e:
            print("Failed to save symptom cache:", e)

    state['category'] = category
    return state

# --- NOTION helper (defensive) ---
def add_appointment_to_notion(state: Dict[str, Any]):
    if not notion or not NOTION_DATABASE_ID:
        print("Notion not configured — skipping save.")
        return

    try:
        meeting_date = datetime.today().date()
        chosen_slot = state.get('chosen_slot', "09:00 AM")
        # parse time from slot (assumes format like "9:00 AM")
        try:
            meeting_start = datetime.strptime(f"{meeting_date} {chosen_slot}", "%Y-%m-%d %I:%M %p")
        except Exception:
            meeting_start = datetime.combine(meeting_date, datetime.now().time())
        meeting_end = meeting_start + timedelta(hours=1)

        notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Name": {"title": [{"text": {"content": state.get("name", "N/A")}}]},
                "Age": {"number": int(state.get("age") or 0)},
                "Email": {"email": state.get("email", "")},
                "Location": {"rich_text": [{"text": {"content": state.get("location", "")}}]},
                "Symptom": {"rich_text": [{"text": {"content": state.get("symptom", "")}}]},
                "Doctor Type": {"rich_text": [{"text": {"content": state.get("doctor_type", "")}}]},
                "Time Slot": {"rich_text": [{"text": {"content": state.get("chosen_slot", "")}}]},
                "Meeting Time": {"date": {
                    "start": meeting_start.isoformat(),
                    "end": meeting_end.isoformat()
                }},
                "Status": {"select": {"name": "Confirmed"}}
            }
        )
        print("✅ Appointment added to Notion")
    except Exception as e:
        print(f"❌ Failed to add appointment to Notion: {e}")

# --- FLASK ROUTES ---
def default_langgraph_state() -> Dict[str, Any]:
    return {
        "name": None,
        "age": None,
        "email": None,
        "location": None,
        "symptom": None,
        "category": None,
        "doctor_type": None,
        "time_slots": [],
        "chosen_slot": None,
        "final_message": None,
        "current_step": "start"
    }

@app.route('/')
def home():
    session.clear()
    session['langgraph_state'] = default_langgraph_state()
    # You need index.html in templates; for now just show a minimal page in templates
    return render_template("index.html")

@app.route('/submit-details', methods=['POST'])
def submit_details():
    s = session.get('langgraph_state', default_langgraph_state())
    # safe getters
    name = request.form.get('name', '').strip()
    age_raw = request.form.get('age', '').strip()
    email = request.form.get('email', '').strip()
    location = request.form.get('location', '').strip()

    # Validate age
    try:
        age_val = int(age_raw) if age_raw != "" else None
    except ValueError:
        flash("Invalid age provided. Please enter a number.")
        return redirect(url_for('home'))

    s['name'] = name or None
    s['age'] = age_val
    s['email'] = email or None
    s['location'] = location or None
    # store back to session
    session['langgraph_state'] = s

    next_path = age_router(s)
    if next_path == "pediatrics":
        s = pediatrics_node(s)
        s = show_time_slots_node(s)
        session['langgraph_state'] = s
        return redirect(url_for('appointment_page'))
    elif next_path == "geriatrics":
        s = geriatrics_node(s)
        s = show_time_slots_node(s)
        session['langgraph_state'] = s
        return redirect(url_for('appointment_page'))
    else:
        s['current_step'] = 'symptom_input'
        session['langgraph_state'] = s
        return redirect(url_for('symptom_page'))

@app.route('/symptom')
def symptom_page():
    s = session.get('langgraph_state', default_langgraph_state())
    if s.get('current_step') != 'symptom_input':
        return redirect(url_for('home'))
    return render_template("symptom.html")

@app.route('/submit-symptom', methods=['POST'])
def submit_symptom():
    s = session.get('langgraph_state', default_langgraph_state())
    symptom_text = request.form.get('symptom', '').strip()
    if not symptom_text:
        flash("Please provide symptom details.")
        return redirect(url_for('symptom_page'))

    s['symptom'] = symptom_text
    s = classify_symptom_node(s)
    next_path = symptom_router(s)

    if next_path == "general":
        s = general_node(s)
        s = show_time_slots_node(s)
        s['current_step'] = 'choose_slot'
        session['langgraph_state'] = s
        return redirect(url_for('appointment_page'))
    elif next_path == "mentalhealth":
        s = mentalhealth_node(s)
        s = show_time_slots_node(s)
        s['current_step'] = 'choose_slot'
        session['langgraph_state'] = s
        return redirect(url_for('appointment_page'))
    elif next_path == "emergency":
        s = emergency_node(s)
        s['current_step'] = 'ambulance_dispatched'
        s['final_message'] = "Emergency detected. Please call local emergency services immediately."
        session['langgraph_state'] = s
        return redirect(url_for('confirm_page'))
    else:
        session['langgraph_state'] = s
        return redirect(url_for('home'))

@app.route('/appointment')
def appointment_page():
    s = session.get('langgraph_state', default_langgraph_state())
    return render_template("appointment.html", time_slots=s.get('time_slots', []))

@app.route('/book-slot', methods=['POST'])
def book_slot():
    s = session.get('langgraph_state', default_langgraph_state())
    slot = request.form.get('slot')
    if not slot:
        flash("Please pick a slot.")
        return redirect(url_for('appointment_page'))
    s['chosen_slot'] = slot
    s = confirm_appointment_node(s)
    # Save to Notion (best-effort)
    add_appointment_to_notion(s)
    session['langgraph_state'] = s
    return redirect(url_for('confirm_page'))

@app.route('/confirm')
def confirm_page():
    s = session.get('langgraph_state', default_langgraph_state())
    return render_template("confirm.html", data=s)

@app.route('/generate-pdf')
def generate_pdf():
    s = session.get('langgraph_state', default_langgraph_state())
    template_path = "OP_template_final.pdf"
    output_path = "confirmation.pdf"

    if not os.path.exists(template_path):
        return "Template PDF not found on server. Place OP_template_final.pdf in project root.", 500

    # Coordinates for each field on the template (x, y) in points
    field_coords = {
        "name": (100, 700),
        "age": (100, 680),
        "email": (100, 660),
        "location": (100, 640),
        "symptom": (100, 620),
        "doctor_type": (100, 600),
        "chosen_slot": (100, 580)
    }

    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont("Helvetica", 12)

    for field, (x, y) in field_coords.items():
        value = s.get(field) or ""
        try:
            value_str = str(value)
        except Exception:
            value_str = ""
        can.drawString(x, y, value_str[:200])  # limit length to avoid overflow

    can.save()
    packet.seek(0)

    # Read the existing template PDF
    try:
        existing_pdf = PdfReader(open(template_path, "rb"))
        overlay_pdf = PdfReader(packet)
        output = PdfWriter()

        page = existing_pdf.pages[0]
        overlay_page = overlay_pdf.pages[0]

        # PyPDF2 has changed names across versions; try both
        if hasattr(page, "merge_page"):
            page.merge_page(overlay_page)
        elif hasattr(page, "mergePage"):
            page.mergePage(overlay_page)
        else:
            # As fallback: try adding both pages into writer (this may not overlay)
            pass

        output.add_page(page)

        with open(output_path, "wb") as outputStream:
            output.write(outputStream)
    except Exception as e:
        print("PDF generation failed:", e)
        return f"PDF generation failed: {e}", 500

    return send_file(output_path, as_attachment=True)

@app.route('/generate-audio')
def generate_audio():
    s = session.get('langgraph_state', default_langgraph_state())
    if not s.get('final_message'):
        flash("No final message to convert to audio.")
        return redirect(url_for('confirm_page'))

    if not ELEVEN_API_KEY or not VOICE_ID:
        return "ElevenLabs API key or Voice ID not configured.", 500

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    data = {
        "text": s['final_message'],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
    except Exception as e:
        print("ElevenLabs request failed:", e)
        return "Failed to contact ElevenLabs API.", 500

    if response.status_code != 200:
        print("ElevenLabs error:", response.status_code, response.text)
        return f"ElevenLabs error: {response.status_code} - {response.text}", 500

    # Validate that the response is audio by checking headers (best-effort)
    ct = response.headers.get("Content-Type", "")
    if "audio" not in ct and not response.content.startswith(b"ID3"):  # ID3: mp3 signature heuristic
        print("ElevenLabs returned non-audio content:", ct)
        return "ElevenLabs returned unexpected content.", 500

    output_file = os.path.join("static", "output.mp3")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(response.content)

    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    # Note: in production use a WSGI server and set debug from env
    debug_flag = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug_flag, port=int(os.getenv("PORT", 5001)))