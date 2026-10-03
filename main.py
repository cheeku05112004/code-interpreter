import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer

    try:
        exec(code, {})

        return {
            "success": True,
            "output": stdout_buffer.getvalue() + stderr_buffer.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def extract_traceback_lines(traceback_text: str) -> List[int]:
    """
    Get the last Python source line mentioned in the traceback.
    """
    matches = re.findall(r'line (\d+)', traceback_text)

    if matches:
        return [int(matches[-1])]

    return []


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Ask AIPipe LLM to identify source error lines.
    Fall back to traceback extraction if AI call fails.
    """

    fallback = extract_traceback_lines(traceback_text)

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        return fallback

    try:
        client = OpenAI(
            api_key=token,
            base_url="https://aipipe.org/openrouter/v1"
        )

        prompt = f"""
Analyze this Python code and its traceback.

Find the exact line number(s) in the ORIGINAL CODE
where the error occurred.

Return ONLY one JSON object in this exact format:
{{"error_lines":[3]}}

Do not include markdown.
Do not include explanations.
Do not include library/internal Python lines.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise Python traceback analyzer. "
                        "Return JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # Remove accidental markdown fences
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        result = ErrorAnalysis.model_validate_json(content)

        return result.error_lines

    except Exception:
        # Never let the AI service failure break our API
        return fallback


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    # Successful execution: AI is NOT called
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error case: analyze with AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
