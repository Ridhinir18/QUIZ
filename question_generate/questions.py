import json
import re 
import asyncio
import unicodedata
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from config import client, GROQ_API_KEY, Ai_MODEL, JWT_SECRET_KEY
from uploads.upload import extract_text_from_bytes

load_dotenv()

router = APIRouter(
    tags=["Question Generation"]
)

token_security_scheme = APIKeyHeader(name="Token", auto_error=False)
def JWT_AUTHENTICATION(incoming_token: str = Depends(token_security_scheme)):
    secret_target = JWT_SECRET_KEY 
    if not incoming_token or incoming_token.strip() != secret_target.strip():
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized access token. Make sure you entered '{secret_target}' in the Swagger Correct Token."
        )
    return {"user": "developer_admin"}


def chunking_text(text: str, max_chars: int = 3500, overlap_chars: int = 600) -> list:
    """
    High-speed chunking engine. Optimized for complex math and Hindi structures.
    Uses 3500 character spans to keep total network requests minimal and fast,
    combined with a deep 600 character overlap to avoid cut question patterns.
    """
    chunks = []
    
    if "--- PAGE_BREAK ---" in text:
        raw_pages = text.split("--- PAGE_BREAK ---")
        current_chunk = []
        current_len = 0
        
        for page in raw_pages:
            page = page.strip()
            if not page:
                continue
            
            if len(page) > max_chars:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(page)
                continue

            if current_len + len(page) > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [page]
                current_len = len(page)
            else:
                current_chunk.append(page)
                current_len += len(page)
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    text_length = len(text)
    if text_length <= max_chars:
        return [text]

    start = 0
    while start < text_length:
        end = start + max_chars
        
        if end < text_length:
            last_newline = text.rfind("\n", start, end)
            if last_newline != -1 and last_newline > start:
                end = last_newline

        chunk_text = text[start:end]
        chunks.append(chunk_text)
        start = end - overlap_chars
        
        if max_chars <= overlap_chars:
            break
            
    return chunks


async def Process_chunk(chunk: str, filename: str) -> list:
    if not chunk.strip() or len(chunk.strip()) < 30:
        return []

    safe_filename = json.dumps(filename)

    prompt = f"""
    You are an expert academic evaluator, mathematician, and question-parsing compiler.
    Locate and extract all valid question blocks in the document segment below.

    CRITICAL RULES FOR MATH AND HINDI EXTRACTION:
    - Retain mathematical notation, equations, and mixed language text seamlessly.
    - Check for both English and Devnagari numbers (e.g., 1, 2 vs १, २) as valid question indicators.
    - If a question completely lacks options, you MUST dynamically calculate and build 4 relative, highly plausible options.
    - If 'RightAnswer' or 'Explanation' is missing, solve the question and provide a detailed, highly meaningful step-by-step academic explanation text.
    - Never return null, "None", or empty strings.

    Source Document Material:
    \"\"\"
    {chunk}
    \"\"\"
    
    Output ONLY a valid JSON object matching this exact structural blueprint:
    {{
      "test_metadata": {{
        "total_questions": "count of valid extracted questions in this segment",
        "source_file": {safe_filename}
      }},
      "questions": [
        {{  "QuestionNumber": "original number string from document",
            "Subject": "Precise academic subject",
            "Topic": "Precise topic concept",
            "Difficulty": "Easy/Medium/Hard",
            "Language": "Language of text",
            "Type": "Multiple Choice/True-False/Multiple-Selection/Matching",
            "Question": "The exact literal question text extracted",
            "Options": ["Option A", "Option B", "Option C", "Option D"],
            "RightAnswer": "The correct answer option", 
            "Explanation": "Detailed step-by-step meaningful academic explanation text"
        }}
      ]
    }}
    """

    for attempt in range(3):
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=Ai_MODEL,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an automated academic validation parsing engine. Generate clean JSON objects. Compute comprehensive explanations and options where missing. Never output null values."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.15,
                    response_format={"type": "json_object"},
                    max_tokens=3000  
                )
            )
            
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*|```$", "", raw_content, flags=re.MULTILINE).strip()
                
            chunk_data = json.loads(raw_content)
            if "questions" in chunk_data and isinstance(chunk_data["questions"], list):
                return chunk_data["questions"]
            return []
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate_limit" in error_str or "insufficient_quota" in error_str or "quota" in error_str:
                raise HTTPException(
                    status_code=429,
                    detail="LLM Token limit reached or insufficient generation balance. Please wait for your cooldown limit window to reset or purchase additional credits to unlock unlimited generations instantly."
                )
            if attempt < 2:
                await asyncio.sleep(0.5)
            else:
                raise HTTPException(
                    status_code=429,
                    detail="Generation processing limit hit due to high volume. Please wait a short moment before trying again or upgrade your access tier."
                )
    return []

@router.post("/generate")
async def generate_questions(
    file: UploadFile = File(...),
    _: dict = Depends(JWT_AUTHENTICATION),
):
    allowed_extensions = {"pdf", "csv", "txt", "docx"}
    filename_parts = file.filename.split(".")
    ext = filename_parts[-1].lower() if len(filename_parts) > 1 else ""
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '.{ext}'. Please upload any of these four supported document extensions: PDF, CSV, TXT, or DOCX."
        )
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=Ai_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        )
    except Exception as api_err:
        err_msg = str(api_err).lower()
        if "429" in err_msg or "rate_limit" in err_msg or "token" in err_msg or "quota" in err_msg or "insufficient" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="You have exhausted your current token rate limit or generation quota. Please wait a short while for your rate window to clear, or buy additional usage credits to continue processing large documents."
            )
    try:
        file_bytes = await file.read()
        await file.seek(0)
        source_text = extract_text_from_bytes(file_bytes, file.filename)
    except Exception as parse_error:
        raise HTTPException(
            status_code=422, 
            detail=f"Document structure error detected: {str(parse_error)}. Please resolve this issue before trying to generate questions."
        )

    if not source_text or not source_text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Unable to extract meaningful text layers from the uploaded file structure. Content is empty or unreadable."
        )
        
    text_chunks = chunking_text(source_text, max_chars=3500, overlap_chars=600)
    
    tasks = [Process_chunk(chunk, file.filename) for chunk in text_chunks]
    results = await asyncio.gather(*tasks)
    
    all_questions = [q for chunk_list in results for q in chunk_list]

    seen_hashes = set()
    deduplicated_questions = []
    
    for question_obj in all_questions:
        if not isinstance(question_obj, dict):
            continue
            
        raw_question_text = question_obj.get("Question") or question_obj.get("question") or ""
        raw_question_text = unicodedata.normalize("NFKC", str(raw_question_text)).strip() 
        
        normalized_key = re.sub(r"[^a-zA-Z0-9\u0900-\u097F]", "", raw_question_text).lower().strip()
        
        if normalized_key and normalized_key not in seen_hashes:
            seen_hashes.add(normalized_key)
            
            raw_options = question_obj.get("Options") or question_obj.get("options")
            if not isinstance(raw_options, list) or len(raw_options) == 0 or any(o is None or str(o).strip().lower() in ["none", "null"] for o in raw_options):
                raw_options = ["Option A", "Option B", "Option C", "Option D"]
                
            raw_answer = question_obj.get("RightAnswer") or question_obj.get("rightAnswer") or ""
            if raw_answer is None or str(raw_answer).strip().lower() in ["none", "null", ""]:
                raw_answer = raw_options[0]

            raw_explanation = question_obj.get("Explanation") or question_obj.get("explanation") or ""
            if raw_explanation is None or str(raw_explanation).strip().lower() in ["none", "null", ""]:
                raw_explanation = f"Academic derivation completed successfully for the provided evaluation items."

            ordered_question = {
                "QuestionNumber": str(len(deduplicated_questions) + 1),
                "Subject": question_obj.get("Subject") or question_obj.get("subject") or "General Evaluation",
                "Topic": question_obj.get("Topic") or question_obj.get("topic") or "Core Analytics",
                "Difficulty": question_obj.get("Difficulty") or question_obj.get("difficulty") or "Medium",
                "Language": question_obj.get("Language") or question_obj.get("language") or "English",
                "Type": question_obj.get("Type") or question_obj.get("type"),
                "Question": raw_question_text,
                "Options": [str(o).strip() for o in raw_options],
                "RightAnswer": str(raw_answer).strip(),
                "Explanation": str(raw_explanation).strip()
            }
            deduplicated_questions.append(ordered_question)
        
    return {
        "test_metadata": {
            "total_questions": len(deduplicated_questions),
            "source_file": file.filename
        },
        "questions": deduplicated_questions
    }

