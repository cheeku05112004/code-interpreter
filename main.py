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


# Enable CORS
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
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer

    try:
        exec(code, {})

        output = (
            stdout_buffer.getvalue()
            + stderr_buffer.getvalue()
        )

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def extract_traceback_lines(traceback_text: str) -> List[int]:
    """
    Extract line numbers from Python traceback.
    """
    matches = re.findall(r'line (\d+)', traceback_text)

    if matches:
        return [int(matches[-1])]

    return []


def analyze_error_with_ai(
    code: str,
    traceback_text: str
) -> List[int]:
    """
    Use AIPipe LLM to identify the source-code error line.
    """

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        return extract_traceback_lines(traceback_text)

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

Your task is to identify the exact line number or line numbers
in the ORIGINAL CODE where the error occurred.

Rules:
1. Return only source-code line numbers.
2. Do not return internal Python library lines.
3. For a runtime error, return the line containing the failing statement.
4. For a syntax error, return the source line indicated by the traceback.
5. Return a JSON object exactly like:
{{"error_lines":[3]}}

ORIGINAL CODE:
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
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        result = ErrorAnalysis.model_validate_json(content)
        return result.error_lines

    except Exception:
        return extract_traceback_lines(traceback_text)


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # No AI call when execution succeeds
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # AI is called only when an error occurs
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
