import os
import traceback
from io import StringIO
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})
        output = sys.stdout.getvalue()
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


def analyze_error_with_ai(code: str, error_traceback: str) -> list[int]:
    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

Identify the exact line number(s) in the USER CODE where the error occurred.
Do NOT use line numbers from main.py or from the traceback.
The first line of USER CODE is line 1.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return ONLY JSON in this format:
{{"error_lines": [3]}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


@app.post("/code-interpreter")
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