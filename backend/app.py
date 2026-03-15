from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import fitz
from openai import AsyncOpenAI
from supabase import create_client, Client
import json, io, os
from datetime import datetime, timezone, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from dotenv import load_dotenv

load_dotenv()

openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MAX_FILE_SIZE = 20 * 1024 * 1024
FREE_WEEKLY_LIMIT = 3


def get_user_from_token(token: str):
    try:
        resp = supabase.auth.get_user(token)
        return resp.user
    except Exception:
        return None


def get_usage_this_week(user_id: str) -> int:
    week_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    resp = supabase.table("usage").select("id").eq("user_id", user_id).gte("created_at", week_start).execute()
    return len(resp.data)


def log_usage(user_id: str):
    supabase.table("usage").insert({"user_id": user_id}).execute()


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n\n".join(p.get_text() for p in doc)[:18000]


@app.get("/me")
async def me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    usage = get_usage_this_week(user.id)
    return {
        "id": user.id,
        "email": user.email,
        "usage": usage,
        "limit": FREE_WEEKLY_LIMIT,
        "can_analyze": usage < FREE_WEEKLY_LIMIT
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    mcq_count: int = Form(10),
    authorization: str = Header(None),
):
    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Usage check
    usage = get_usage_this_week(user.id)
    if usage >= FREE_WEEKLY_LIMIT:
        raise HTTPException(status_code=402, detail=f"Weekly limit reached. You've used {usage}/{FREE_WEEKLY_LIMIT} PDFs this week. Upgrade to Pro for unlimited access.")

    # File validation
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail=f"Only PDF files are allowed. Got: {ext or 'unknown'}")
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF.")

    text = extract_text(pdf_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF. It may be scanned or image-based.")

    prompt = f"""You are an expert study assistant for students. Analyze this PDF content deeply and extract everything a student needs to study and memorize.

PDF Content:
{text}

Return a single raw JSON object (no markdown, no code fences) with this exact structure:
{{
  "summary": "A clear 3-4 sentence overview of what this document covers",
  "priority_topics": [
    {{"topic": "Topic name", "why": "Why this is important to study", "weight": "high/medium"}}
  ],
  "definitions": [
    {{"term": "Term or concept", "definition": "Clear, memorizable definition"}}
  ],
  "formulas": [
    {{"name": "Formula name", "formula": "The formula itself", "variables": "What each variable means", "usage": "When/how to use it"}}
  ],
  "flashcards": [
    {{"front": "Question or prompt", "back": "Answer or explanation", "mnemonic": "Memory trick or empty string"}}
  ],
  "notes": {{
    "summary": "Same overview",
    "concepts": ["Concept: definition"],
    "points": ["Key takeaway"]
  }},
  "mcqs": [
    {{"question": "Question?", "options": ["A", "B", "C", "D"], "correct_index": 0, "explanation": "Why correct"}}
  ]
}}

Rules:
- priority_topics: 3-6 most exam-important topics
- definitions: ALL defined terms, minimum 5
- formulas: ALL formulas. Empty array [] if none
- flashcards: 8-12 cards with creative mnemonics
- mcqs: exactly {mcq_count} questions
- Return ONLY the JSON"""

    response = await openai_client.chat.completions.create(
        model="openrouter/auto",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try again.")

    log_usage(user.id)
    new_usage = usage + 1

    return {"result": result, "usage": new_usage, "limit": FREE_WEEKLY_LIMIT}


# ── PDF generation ────────────────────────────────────────────────────────────
INK       = colors.HexColor("#1a1a2e")
INK2      = colors.HexColor("#4a4a6a")
ACCENT    = colors.HexColor("#2563eb")
ACCENT_BG = colors.HexColor("#eff6ff")
GREEN     = colors.HexColor("#16a34a")
GREEN_BG  = colors.HexColor("#f0fdf4")
AMBER     = colors.HexColor("#d97706")
AMBER_BG  = colors.HexColor("#fffbeb")
RED       = colors.HexColor("#dc2626")
RED_BG    = colors.HexColor("#fef2f2")
PURPLE    = colors.HexColor("#7c3aed")
RULE      = colors.HexColor("#e2e8f0")


def sanitize(text: str) -> str:
    """Replace unicode symbols unsupported by ReportLab built-in fonts."""
    replacements = {
        '×': 'x',
        '·': '.',
        '÷': '/',
        '±': '+/-',
        '≈': '~',
        '≠': '!=',
        '≤': '<=',
        '≥': '>=',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '∞': 'inf',
        '∑': 'sum',
        '∏': 'prod',
        '√': 'sqrt',
        '∂': 'd',
        '∫': 'integral',
        '∆': 'delta',
        'Δ': 'delta',
        'Ω': 'Omega',
        'ω': 'omega',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'θ': 'theta',
        'λ': 'lambda',
        'μ': 'mu',
        'π': 'pi',
        'σ': 'sigma',
        'τ': 'tau',
        'φ': 'phi',
        'ψ': 'psi',
        '−': '-',  # minus sign
        '–': '-',  # en dash
        '—': '--', # em dash
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '“': '"',  # left double quote
        '”': '"',  # right double quote
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def make_styles():
    return {
        "doc_title": ParagraphStyle("doc_title", fontName="Helvetica-Bold", fontSize=26, textColor=INK, spaceAfter=4, leading=32),
        "doc_sub": ParagraphStyle("doc_sub", fontName="Helvetica", fontSize=12, textColor=INK2, spaceAfter=20, leading=16),
        "section_label": ParagraphStyle("section_label", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT, spaceBefore=22, spaceAfter=8, leading=12),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15, textColor=INK, spaceBefore=18, spaceAfter=6, leading=20),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=11, textColor=INK2, spaceAfter=5, leading=17),
        "body_bold": ParagraphStyle("body_bold", fontName="Helvetica-Bold", fontSize=11, textColor=INK, spaceAfter=5, leading=17),
        "mono": ParagraphStyle("mono", fontName="Courier-Bold", fontSize=12, textColor=ACCENT, spaceAfter=4, leading=16),
        "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=9, textColor=INK2, spaceAfter=3, leading=13),
        "card_q": ParagraphStyle("card_q", fontName="Helvetica-Bold", fontSize=11, textColor=INK, leading=16),
        "card_a": ParagraphStyle("card_a", fontName="Helvetica", fontSize=10, textColor=INK2, leading=15),
        "card_m": ParagraphStyle("card_m", fontName="Helvetica-Oblique", fontSize=9, textColor=PURPLE, leading=13),
        "mcq_q": ParagraphStyle("mcq_q", fontName="Helvetica-Bold", fontSize=11, textColor=INK, spaceAfter=6, leading=16),
        "mcq_correct": ParagraphStyle("mcq_correct", fontName="Helvetica-Bold", fontSize=10, textColor=GREEN, spaceAfter=3, leading=14),
        "mcq_wrong": ParagraphStyle("mcq_wrong", fontName="Helvetica", fontSize=10, textColor=INK2, spaceAfter=3, leading=14),
        "mcq_expl": ParagraphStyle("mcq_expl", fontName="Helvetica-Oblique", fontSize=9, textColor=GREEN, spaceAfter=0, leading=13),
        "pill_high": ParagraphStyle("pill_high", fontName="Helvetica-Bold", fontSize=8, textColor=RED, leading=10),
        "pill_med": ParagraphStyle("pill_med", fontName="Helvetica-Bold", fontSize=8, textColor=AMBER, leading=10),
    }


def section_header(label, s):
    return [HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=8), Paragraph(label.upper(), s["section_label"])]


@app.post("/download")
async def download(data: dict, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    s = make_styles()
    story = []

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("NOITAR NOTES", s["section_label"]))
    story.append(Paragraph(data.get("filename", "Document"), s["doc_title"]))
    story.append(Paragraph("AI-generated study notes", s["doc_sub"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=20))

    if data.get("summary"):
        story += section_header("Overview", s)
        story.append(Paragraph(data["summary"], s["body"]))

    topics = data.get("priority_topics", [])
    if topics:
        story += section_header("Priority Topics", s)
        for t in topics:
            w = t.get("weight", "medium")
            pill_style = s["pill_high"] if w == "high" else s["pill_med"]
            pill_bg = RED_BG if w == "high" else AMBER_BG
            pill_color = RED if w == "high" else AMBER
            pill = Table([[Paragraph("HIGH" if w == "high" else "MED", pill_style)]], colWidths=[40])
            pill.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),pill_bg),("BOX",(0,0),(-1,-1),0.5,pill_color),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
            row = Table([[pill, Paragraph(f"<b>{t['topic']}</b> — {t.get('why','')}", s["body"])]], colWidths=[50, CONTENT_W-50])
            row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            story.append(row)

    defs = data.get("definitions", [])
    if defs:
        story += section_header("Definitions", s)
        rows = [[Paragraph(sanitize(d.get("term","")), s["body_bold"]), Paragraph(sanitize(d.get("definition","")), s["body"])] for d in defs]
        tbl = Table(rows, colWidths=[CONTENT_W*0.30, CONTENT_W*0.70])
        tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("LINEBELOW",(0,0),(-1,-2),0.4,RULE),("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fafafa")),("BACKGROUND",(0,0),(0,-1),ACCENT_BG),("BOX",(0,0),(-1,-1),0.5,RULE)]))
        story.append(tbl)

    formulas = data.get("formulas", [])
    if formulas:
        story += section_header("Formulas", s)
        for f in formulas:
            inner = [[Paragraph(sanitize(f.get("name","")), s["body_bold"]), Paragraph(sanitize(f.get("formula","")), s["mono"])]]
            if f.get("variables"): inner.append([Paragraph("Variables", s["caption"]), Paragraph(sanitize(f["variables"]), s["caption"])])
            if f.get("usage"): inner.append([Paragraph("Usage", s["caption"]), Paragraph(sanitize(f["usage"]), s["caption"])])
            tbl = Table(inner, colWidths=[CONTENT_W*0.25, CONTENT_W*0.75])
            tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),ACCENT_BG),("BOX",(0,0),(-1,-1),1,ACCENT),("LINEBELOW",(0,0),(-1,-2),0.4,colors.HexColor("#bfdbfe")),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(tbl)
            story.append(Spacer(1, 8))

    notes = data.get("notes", {})
    if notes:
        story += section_header("Study Notes", s)
        if notes.get("summary"): story.append(Paragraph(sanitize(notes["summary"]), s["body"]))
        if notes.get("concepts"):
            story.append(Paragraph("Key Concepts", s["h2"]))
            for c in notes["concepts"]: story.append(Paragraph(sanitize(f"• {c}"), s["body"]))
        if notes.get("points"):
            story.append(Paragraph("Important Points", s["h2"]))
            for p in notes["points"]: story.append(Paragraph(sanitize(f"• {p}"), s["body"]))

    flashcards = data.get("flashcards", [])
    if flashcards:
        story.append(PageBreak())
        story += section_header("Flashcards", s)
        CARD_W = (CONTENT_W - 12) / 2
        pairs = [flashcards[i:i+2] for i in range(0, len(flashcards), 2)]
        for pair in pairs:
            cells = []
            for fc in pair:
                inner_rows = [[Paragraph("QUESTION", s["section_label"])],[Paragraph(fc.get("front",""), s["card_q"])],[Spacer(1,6)],[HRFlowable(width="100%", thickness=0.5, color=RULE)],[Spacer(1,4)],[Paragraph("ANSWER", s["section_label"])],[Paragraph(fc.get("back",""), s["card_a"])]]
                if fc.get("mnemonic"): inner_rows += [[Spacer(1,4)],[Paragraph(f"Mnemonic: {fc['mnemonic']}", s["card_m"])]]
                inner_tbl = Table(inner_rows, colWidths=[CARD_W-24])
                inner_tbl.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),("VALIGN",(0,0),(-1,-1),"TOP")]))
                cells.append(inner_tbl)
            if len(cells) == 1: cells.append(Paragraph("", s["body"]))
            row_tbl = Table([cells], colWidths=[CARD_W, CARD_W])
            row_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fafafa")),("BOX",(0,0),(0,-1),0.8,PURPLE),("BOX",(1,0),(1,-1),0.8,PURPLE),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(row_tbl)
            story.append(Spacer(1, 10))

    mcqs = data.get("mcqs", [])
    if mcqs:
        story.append(PageBreak())
        story += section_header("Multiple Choice Questions", s)
        for i, q in enumerate(mcqs):
            items = [Paragraph(f"Q{i+1}.  {q['question']}", s["mcq_q"])]
            for j, opt in enumerate(q["options"]):
                is_correct = j == q.get("correct_index", -1)
                items.append(Paragraph(("✓  " if is_correct else f"{'ABCD'[j]}.  ") + opt, s["mcq_correct"] if is_correct else s["mcq_wrong"]))
            if q.get("explanation"):
                items += [Spacer(1,3), Paragraph(f"↳ {q['explanation']}", s["mcq_expl"])]
            block = Table([[item] for item in items], colWidths=[CONTENT_W-24])
            block.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fafafa")),("BOX",(0,0),(-1,-1),0.5,RULE),("LINEBEFORE",(0,0),(0,-1),3,ACCENT),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(KeepTogether(block))
            story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=noitar-notes.pdf"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
