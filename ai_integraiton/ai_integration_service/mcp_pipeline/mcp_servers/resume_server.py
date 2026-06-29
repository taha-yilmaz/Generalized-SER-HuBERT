"""
MCP Server - Stream 1: Resume Analysis
=======================================
Adapted from the project's original implementation.
Mimari: pymupdf4llm + pdfplumber (Hybrid) → Gemini 2.5 Flash → Pydantic Schema
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ResumeAnalysisServer")


# --- Resume parsing functions ---

def extract_hybrid_content(pdf_path: str) -> dict:
    """
    Reads PDF using TWO methods to prevent data loss:
    1. Markdown: Preserves structure and columns.
    2. Raw Text: Backup for missing words.
    """
    extracted_data = {"markdown": "", "raw_text": ""}

    def fix_encoding(text: str) -> str:
        """Repair mojibake, double-encoding, and LaTeX ligature artifacts."""
        if not text:
            return text

        # 1. ftfy: çift-encoding, mojibake, Windows-1252 karışımını otomatik düzelt
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            print("⚠️ ftfy not installed, falling back to manual fix")
            try:
                repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
                if "Ã" in text or "Â" in text:
                    text = repaired
            except Exception:
                pass

        # 2. LaTeX diacritic ligatures (boşluklu ve boşluksuz formlar)
        import re
        latex_patterns = [
            (r"¨\s*u", "ü"), (r"¨\s*U", "Ü"),
            (r"¨\s*o", "ö"), (r"¨\s*O", "Ö"),
            (r"¨\s*i", "ï"), (r"¨\s*I", "İ"),
            (r"˘\s*g", "ğ"), (r"˘\s*G", "Ğ"),
            (r"¸\s*s", "ş"), (r"¸\s*S", "Ş"),
            (r"¸\s*c", "ç"), (r"¸\s*C", "Ç"),
            (r"´\s*c", "ć"), (r"´\s*C", "Ć"),
        ]
        for pattern, replacement in latex_patterns:
            text = re.sub(pattern, replacement, text)

        return text

    # Method 1: Markdown
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(pdf_path)
        extracted_data["markdown"] = fix_encoding(md)
    except Exception as error:
        print(f"⚠️ Markdown conversion error: {error}")
        extracted_data["markdown"] = ""

    # Method 2: Raw text
    try:
        import pdfplumber
        raw_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, use_text_flow=True)
                if page_text:
                    raw_text += page_text + "\n"
        extracted_data["raw_text"] = fix_encoding(raw_text)
    except Exception as error:
        print(f"⚠️ Raw text extraction error: {error}")
        extracted_data["raw_text"] = ""

    print(f"[extract_hybrid DEBUG] raw_text first 200: {repr(extracted_data['raw_text'][:200])}")
    print(f"[extract_hybrid DEBUG] markdown first 200: {repr(extracted_data['markdown'][:200])}")
    return extracted_data


def analyze_resume_hybrid(pdf_content: dict, api_key: str = "", model_name: str = "gemma4:e4b") -> dict | None:
    """Analyze resume using local Ollama (Gemma 3 4B) with hybrid extraction + Pydantic validation."""
    from llm.ollama_client import generate_json
    from pydantic import BaseModel, ValidationError
    from typing import List, Dict

    # Structured item schemas — used by CV builder for direct form auto-fill
    class EducationItem(BaseModel):
        school_name: str = ""
        degree: str = ""           # Bachelor's | Master's | PhD | Bootcamp / Certificate
        field_of_study: str = ""
        start_date: str = ""       # "YYYY-MM"
        end_date: str = ""         # "YYYY-MM" or "" for ongoing

    class ExperienceItem(BaseModel):
        company_name: str = ""
        title: str = ""
        description: str = ""
        start_date: str = ""
        end_date: str = ""
        is_current: bool = False

    class LanguageItem(BaseModel):
        name: str = ""
        level: str = ""            # CEFR: A1|A2|B1|B2|C1|C2; "" → backend defaults to A2

    class ResumeSchema(BaseModel):
        full_name: str = "Not specified"
        contact_information: Dict[str, str] = {}
        education: List[str] = []                       # legacy flat strings (kept for score_resume)
        work_experience: List[str] = []                 # legacy flat strings
        technical_skills: List[str] = []
        educations: List[EducationItem] = []            # structured for CV builder auto-fill
        work_experiences: List[ExperienceItem] = []     # structured
        languages: List[LanguageItem] = []              # structured (name + CEFR level)

    markdown_content = pdf_content.get("markdown", "")
    raw_content = pdf_content.get("raw_text", "")

    prompt = f"""You are an expert Human Resources (HR) analysis agent.
I will provide you with the same resume in two different formats. The goal is to produce the most accurate analysis without data loss.

SOURCE 1 (MARKDOWN FORMAT):
Use this version to understand **headings, lists, and column structures**.
---
{markdown_content}
---

SOURCE 2 (RAW TEXT FORMAT):
Use this version to verify **characters or words that may have been lost** during Markdown conversion.
---
{raw_content}
---

TASK:
By synthesizing both sources, return ONLY a JSON object with this exact schema:
{{
  "full_name": "Candidate's full name",
  "contact_information": {{"email": "...", "phone": "...", "linkedin": "...", "location": "..."}},
  "education": ["School name, department, degree, and graduation year"],
  "work_experience": ["Company name, position, dates, and brief job summary"],
  "technical_skills": ["individual technical skill"],
  "educations": [
    {{"school_name": "...", "degree": "Bachelor's", "field_of_study": "...", "start_date": "YYYY-MM", "end_date": "YYYY-MM"}}
  ],
  "work_experiences": [
    {{"company_name": "...", "title": "...", "description": "...", "start_date": "YYYY-MM", "end_date": "YYYY-MM", "is_current": false}}
  ],
  "languages": [
    {{"name": "English", "level": "C1"}}
  ]
}}

FORMAT REQUIREMENTS:
- "full_name": single string with the candidate's complete name
- "contact_information": a JSON object mapping contact type (email, phone, linkedin, location, etc.) to its value as strings
- "education" (legacy): an array where EACH element is ONE education record as a single string (school + department + degree + year combined). Keep populating this for backward compatibility.
- "work_experience" (legacy): an array where EACH element is ONE job as a single string (company + position + dates + summary combined). Keep populating this for backward compatibility.
- "technical_skills": an array where EACH element is ONE individual skill as a string
- "educations" (structured): One object per education record. Field rules:
  * "school_name": university or institution name
  * "degree": MUST be exactly one of: "Bachelor's", "Master's", "PhD", "Bootcamp / Certificate".
    Mapping examples: BSc/BS/BA/Lisans → "Bachelor's"; MSc/MS/MA/Yüksek Lisans → "Master's"; Doctorate/Doktora → "PhD"; bootcamp/course/certificate/sertifika/kurs → "Bootcamp / Certificate". If unclear, default to "Bachelor's".
  * "field_of_study": department or major (e.g. "Computer Engineering")
  * "start_date" / "end_date": "YYYY-MM". If only year is known, use "YYYY-01". If ongoing or "Present", set "end_date" to "".
- "work_experiences" (structured): One object per job/internship/project role. Field rules:
  * "company_name": employer (or institution for internships/projects)
  * "title": job title
  * "description": one-paragraph summary of responsibilities (concise)
  * "start_date" / "end_date": "YYYY-MM" format. If only year is known, use "YYYY-01".
  * "is_current": true if the role is ongoing ("Present"); when true, "end_date" MUST be "".
- "languages": One object per spoken/written language. Field rules:
  * "name": language name (e.g. "English", "Turkish", "German")
  * "level": MUST be exactly one of "A1", "A2", "B1", "B2", "C1", "C2", or "" if not stated.
    Mapping examples: Native/Anadil/Mother tongue → "C2"; Fluent/Akıcı/Proficient → "C1"; Advanced/İleri/Upper-intermediate → "B2"; Intermediate/Orta → "B1"; Basic/Beginner/Elementary/Temel → "A2"; If only the language name is mentioned with no level → "".

RULES:
1. Extract EVERY single work experience, internship, and project separately.
2. If the resume contains 2 internships, both "work_experience" and "work_experiences" arrays MUST contain 2 entries.
3. Do not merge, summarize across roles, or skip any entry.
4. Count each "Experience" or "Internship" heading as one entry.
5. Extract only information explicitly present in the resume.
6. If a field is missing in the structured objects, use "" (empty string) — except "is_current" which defaults to false.
7. Never fabricate information.
8. Return JSON only — no markdown fences, no explanations."""

    print("🤖 Ollama (Gemma 3 4B) is performing hybrid resume analysis...")

    try:
        raw_result = generate_json(prompt, temperature=0.0, max_tokens=4096)
        if not raw_result:
            print("❌ Ollama returned empty result.")
            return None

        # Drop malformed items inside structured arrays before validation
        def _clean_item_list(items, model_cls):
            if not isinstance(items, list):
                return []
            cleaned = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                try:
                    cleaned.append(model_cls(**it).model_dump())
                except ValidationError:
                    continue
            return cleaned

        if "educations" in raw_result:
            raw_result["educations"] = _clean_item_list(raw_result.get("educations"), EducationItem)
        if "work_experiences" in raw_result:
            raw_result["work_experiences"] = _clean_item_list(raw_result.get("work_experiences"), ExperienceItem)
        if "languages" in raw_result:
            raw_result["languages"] = _clean_item_list(raw_result.get("languages"), LanguageItem)

        # Pydantic validation — eksik/yanlış alanları default ile doldurur
        try:
            validated = ResumeSchema(**raw_result)
            return _with_flat_fallbacks(validated.model_dump())
        except ValidationError as ve:
            print(f"⚠️ Schema validation failed, attempting repair: {ve}")
            # Repair: tip dönüşümleri dene
            repaired = dict(raw_result)

            # List alanları string gelmişse listeye çevir
            for field in ("education", "work_experience", "technical_skills"):
                val = repaired.get(field)
                if val is None:
                    repaired[field] = []
                elif isinstance(val, str):
                    repaired[field] = [val] if val.strip() else []
                elif not isinstance(val, list):
                    repaired[field] = [str(val)]
                else:
                    # Liste ama elemanları string değilse stringleştir
                    repaired[field] = [str(x) for x in val if x is not None]

            # Structured listeler — yanlış tipte gelmişse boş liste
            for field in ("educations", "work_experiences", "languages"):
                if not isinstance(repaired.get(field), list):
                    repaired[field] = []

            # contact_information dict değilse sıfırla
            if not isinstance(repaired.get("contact_information"), dict):
                repaired["contact_information"] = {}
            else:
                # Değerleri string olmayabilir, stringleştir
                repaired["contact_information"] = {
                    str(k): str(v) for k, v in repaired["contact_information"].items()
                }

            # full_name string değilse stringleştir
            if not isinstance(repaired.get("full_name"), str):
                repaired["full_name"] = str(repaired.get("full_name", "Not specified"))

            # Tekrar validate et
            try:
                validated = ResumeSchema(**repaired)
                return _with_flat_fallbacks(validated.model_dump())
            except ValidationError as ve2:
                print(f"❌ Repair failed: {ve2}")
                return None

    except Exception as error:
        print(f"❌ Analysis error: {error}")
        return None


def _with_flat_fallbacks(result: dict) -> dict:
    """If LLM populated structured lists but skipped flat strings, derive flat strings programmatically.
    Keeps the legacy `education`/`work_experience` arrays populated for downstream consumers (score_resume)."""
    if not result.get("education") and result.get("educations"):
        derived = []
        for e in result["educations"]:
            parts = [p for p in [
                e.get("school_name", "").strip(),
                e.get("degree", "").strip(),
                e.get("field_of_study", "").strip(),
            ] if p]
            dates = []
            if e.get("start_date"): dates.append(e["start_date"])
            if e.get("end_date"):   dates.append(e["end_date"])
            else:                   dates.append("Present") if e.get("start_date") else None
            line = ", ".join(parts)
            if dates:
                line = f"{line} ({' - '.join(dates)})"
            if line.strip():
                derived.append(line)
        result["education"] = derived

    if not result.get("work_experience") and result.get("work_experiences"):
        derived = []
        for w in result["work_experiences"]:
            parts = [p for p in [
                w.get("company_name", "").strip(),
                w.get("title", "").strip(),
            ] if p]
            dates = []
            if w.get("start_date"): dates.append(w["start_date"])
            if w.get("is_current"): dates.append("Present")
            elif w.get("end_date"): dates.append(w["end_date"])
            line = ", ".join(parts)
            if dates:
                line = f"{line} ({' - '.join(dates)})"
            if w.get("description"):
                line = f"{line}. {w['description']}"
            if line.strip():
                derived.append(line)
        result["work_experience"] = derived
    return result


# ─── MCP Tool'lar ───

@mcp.tool()
def parse_resume(pdf_path: str, gemini_api_key: str, model_name: str = "gemma4:e4b") -> dict:
    """
    PDF CV'yi parse edip yapılandırılmış profil döndürür.

    Args:
        pdf_path: Resume PDF path
        gemini_api_key: Google Gemini API key
        model_name: Gemini model name

    Returns:
        Structured resume data (full_name, contact, education, experience, skills)
    """
    print(f"📄 Parsing PDF: {pdf_path}")

    # 1. Hybrid extraction (original implementation
    pdf_content = extract_hybrid_content(pdf_path)

    if not pdf_content["markdown"] and not pdf_content["raw_text"]:
        return {"error": "Cannot read PDF file."}

    # 2. Gemini analysis (original implementation
    result = analyze_resume_hybrid(pdf_content, gemini_api_key, model_name)

    if result is None:
        return {"error": "Gemini analysis failed.", "raw_text": pdf_content["raw_text"][:2000]}

    # Include raw_text (usable in fusion)
    result["raw_text"] = pdf_content["raw_text"][:5000]

    print(f"✅ Resume parsed: {result.get('full_name', 'Unknown')}")
    print(f"[parse_resume DEBUG] raw_text first 200: {repr(result['raw_text'][:200])}")
    print(f"[parse_resume DEBUG] full_name: {repr(result.get('full_name'))}")
    print(f"[parse_resume DEBUG] education: {repr(result.get('education'))}")
    print(f"[parse_resume DEBUG] work_experience count: {len(result.get('work_experience', []))}")
    print(f"[parse_resume DEBUG] educations (structured): {len(result.get('educations', []))}")
    print(f"[parse_resume DEBUG] work_experiences (structured): {len(result.get('work_experiences', []))}")
    print(f"[parse_resume DEBUG] languages (structured): {len(result.get('languages', []))}")
    return result


@mcp.tool()
def score_resume(resume_data: dict, job_requirements: str, gemini_api_key: str = "") -> dict:
    """Evaluate resume against job requirements using local Ollama."""
    from llm.ollama_client import generate_json
    import json

    skills_list = resume_data.get("technical_skills", [])
    req_text = job_requirements.strip() if job_requirements else ""

    prompt = f"""Evaluate this resume against the job requirements.

CV:
{json.dumps(resume_data, ensure_ascii=False)[:4000]}

Requirements:
{req_text[:2000] if req_text else "(No explicit requirements provided — infer the target role from the CV context.)"}

Return ONLY this JSON (no markdown):
{{
  "relevance_score": 0.0-1.0,
  "strengths": ["..."],
  "gaps": ["..."],
  "summary": "...",
  "skill_relevance": {{{", ".join(f'"{s}": 0.0' for s in skills_list[:20])}}}
}}

Rules for skill_relevance:
- Rate each listed skill 0.0-1.0 based on how relevant it is to the job/role.
- If job requirements are given, rate against them directly.
- If no requirements given, infer the target role from the CV and rate accordingly.
- 1.0 = core requirement, 0.7 = strongly relevant, 0.5 = somewhat relevant, 0.3 = peripheral, 0.1 = unrelated.
Note: Dates in CV are NOT in the future. Do not take dates as a gap."""

    try:
        result = generate_json(prompt, temperature=0.1, max_tokens=1536)
        if not result or "relevance_score" not in result:
            return {"relevance_score": 0.5, "strengths": [], "gaps": [], "summary": "Scoring failed", "skill_relevance": {}}
        if "skill_relevance" not in result or not isinstance(result.get("skill_relevance"), dict):
            result["skill_relevance"] = {}
        return result
    except Exception as e:
        print(f"⚠️ Score error: {e}")
        return {"relevance_score": 0.5, "strengths": [], "gaps": [], "summary": "Scoring failed", "skill_relevance": {}}

if __name__ == "__main__":
    print("🚀 Resume Analysis MCP Server starting...")
    mcp.run(transport="stdio")
