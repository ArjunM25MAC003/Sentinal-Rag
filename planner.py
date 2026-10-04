import os
import json
from typing import Dict, Any, List
from groq import Groq

# Assuming safe_json_parse is available in sentinelrag.utils
try:
    from sentinelrag.utils import safe_json_parse
except ImportError:
    # Fallback if utils isn't set up yet
    def safe_json_parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract json block if wrapped in markdown
            if "```json" in text:
                try:
                    block = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(block)
                except Exception:
                    pass
            return {}

def planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner node: Decomposes the original user query into 3-5 sub-questions.

    This node uses the Groq API (openai/gpt-oss-20b) to act as a 
    research question decomposer. It takes the main user query and breaks 
    it down into smaller, more manageable sub-questions to improve 
    retrieval across different sources.

    Args:
        state (Dict[str, Any]): The current graph state, which must contain 
            at least the 'question' key.

    Returns:
        Dict[str, Any]: Updated state containing the 'sub_questions' list.
    """
    question = state.get("question", "")
    if not question:
        print("[Planner] No question found in state.")
        return {"sub_questions": []}

    print(f"[Planner] Generating sub-questions for: '{question}'")

    try:
        # Initialize Groq client
        # API key should be in environment variable GROQ_API_KEY
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        system_prompt = (
            "You are an expert research question decomposer. "
            "Your task is to take a complex user question and break it down into "
            "3 to 5 smaller, specific sub-questions that would help answer the main question. "
            "Return ONLY a JSON object in the following format: "
            '{"sub_questions": ["question 1", "question 2", ...]}'
        )

        user_prompt = f"Decompose this question: {question}"

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.3, # Low temperature for more focused output
            max_tokens=500
        )

        content = response.choices[0].message.content
        parsed_json = safe_json_parse(content)
        
        sub_questions = parsed_json.get("sub_questions", [])

        if not sub_questions:
            print("[Planner] Warning: JSON parse returned empty sub_questions. Falling back to original question.")
            sub_questions = [question]

    except Exception as e:
        print(f"[Planner] Error during API call or parsing: {e}")
        # Fallback to the original question if planning fails
        sub_questions = [question]

    print(f"[Planner] Generated {len(sub_questions)} sub-questions: {sub_questions}")
    return {"sub_questions": sub_questions}
