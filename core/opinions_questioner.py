import csv
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from core.extractors import *
from openai import OpenAI
import anthropic


def gemini_parser(gemini_client, model, input_prompt, first_job_prompt, second_job_prompt, third_prompt, gemini_response, gpt_response, claude_response, system_prompt):

    the_prompt = input_prompt + "\n" + f"Your response: {gemini_response}" + "\n" + f"GPT response: {gpt_response}" + "\n" + f"claude response: {claude_response}"

    response = gemini_client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=[the_prompt, third_prompt],
    )

    decision, choice, reasoning, why = extract_changes(response.candidates[0].content.parts[0].text)

    return {"decision": decision,
            "choice": choice,
            "reasoning": reasoning,
            "explanation": why}


def gpt_parser(gpt_client, model, input_prompt, first_job_prompt, second_job_prompt, third_prompt, gemini_response, gpt_response, claude_response, system_prompt):

    the_prompt = input_prompt + "\n" + f"Your response: {gemini_response}" + "\n" + f"GPT response: {gpt_response}" + "\n" + f"claude response: {claude_response}"

    response = gpt_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": the_prompt},
            {"role": "user", "content": f"{third_prompt}"}
        ]
    )

    decision, choice, reasoning, why = extract_changes(response.choices[0].message.content)

    return {"decision": decision,
            "choice": choice,
            "reasoning": reasoning,
            "explanation": why}

def claude_parser(claude_client, model, input_prompt, first_job_prompt, second_job_prompt, third_prompt, gemini_response, gpt_response, claude_response, system_prompt):

    the_prompt = input_prompt + "\n" + f"Your response: {gemini_response}" + "\n" + f"GPT response: {gpt_response}" + "\n" + f"claude response: {claude_response}"

    response = claude_client.messages.create(
        max_tokens=500,
        model=model,
        system=system_prompt,
        messages=[
            {"role": "user", "content": the_prompt},
            {"role": "user","content": third_prompt},
        ]
    )

    decision, choice, reasoning, why = extract_changes(response.content[0].text)

    return {"decision": decision,
            "choice": choice,
            "reasoning": reasoning,
            "explanation": why}

def client():
    load_dotenv()

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    gpt_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))

    df = pd.read_csv("../dataset/v4_atomic_all.csv")

    df_responses = pd.read_csv("../responses/llm_responses.csv")

    sample_response = df_responses.sample(n=1)

    ids = set()

    with open("../responses/llm_responses.csv", mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row['id'])

    with open('../prompts/system_prompt', 'r', encoding='utf-8') as f_in:
        system_instructions = f_in.read()

    with open('../prompts/first_job_prompt', 'r', encoding='utf-8') as f_in:
        first_job_instructions = f_in.read()

    with open('../prompts/second_job_prompt', 'r', encoding='utf-8') as f_in:
        second_job_instructions = f_in.read()

    with open('../prompts/optional_prompt', 'r', encoding='utf-8') as f_in:
        third_prompt = f_in.read()

    with open('../prompts/entry_prompt', 'r', encoding='utf-8') as f_in:
        input_prompt = f_in.read()

    for idx, response_row in sample_response.iterrows():
        entry_id = response_row['id']

        atomic_entry = df.iloc[entry_id]

        json_content = atomic_entry.to_json(orient="records")

        gemini_response = [response_row["gemini_mti"], response_row["gemini_r"], response_row["gemini_explanation"]]
        gpt_response = [response_row["gpt_mti"], response_row["gpt_r"], response_row["gpt_explanation"]]
        claude_response = [response_row["claude_mti"], response_row["claude_r"], response_row["claude_explanation"]]

        gemini_result = gemini_parser(
            gemini_client=gemini_client,
            model="gemini-2.5-flash",
            input_prompt=input_prompt + json_content,
            first_job_prompt=first_job_instructions,
            second_job_prompt=second_job_instructions,
            third_prompt=third_prompt,
            gemini_response=gemini_response,
            gpt_response=gpt_response,
            claude_response=claude_response,
            system_prompt=system_instructions
        )
        
        gpt_result = gpt_parser(
            gpt_client=gpt_client,
            model="gpt-4o-mini",
            input_prompt=input_prompt + json_content,
            first_job_prompt=first_job_instructions,
            second_job_prompt=second_job_instructions,
            third_prompt=third_prompt,
            gemini_response=gemini_response,
            gpt_response=gpt_response,
            claude_response=claude_response,
            system_prompt=system_instructions
        )

        claude_result = claude_parser(claude_client=claude_client, model="claude-sonnet-4-20250514",
            input_prompt=input_prompt + json_content,
            first_job_prompt=first_job_instructions,
            second_job_prompt=second_job_instructions,
            third_prompt=third_prompt,
            gemini_response=gemini_response,
            gpt_response=gpt_response,
            claude_response=claude_response,
            system_prompt=system_instructions
        )

        entry_row = [entry_id, gpt_result, gemini_result, claude_result]
        with open("../responses/changing_opinions.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(entry_row)

if __name__ == "__main__":
    client()
