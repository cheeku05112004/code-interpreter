import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

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

    except BaseException:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def extract_traceback_lines(traceback_text: str) -> List[int]:
    matches = re.findall(r'line (\d+)', traceback_text)
    return [int(x) for x in matches]


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    token = os.environ.get("AIPIPE_TOKEN")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the Python code and traceback below.

Identify the exact source-code line number(s) where the error occurred.

Return JSON in exactly this format:
{{"error_lines": [3]}}

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a precise Python traceback analyzer. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        result = ErrorAnalysis.model_validate_json(content)
        return result.error_lines
    except Exception:
        return extract_traceback_lines(traceback_text)[-1:]


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
