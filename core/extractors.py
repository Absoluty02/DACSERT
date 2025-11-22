import re

def manage_llm_response(response, name):

    first_job_block, second_job_block = split_jobs_from_response(response)

    inference, why, used, explanation = extract_first_job_content_only_used(first_job_block)

    csr_dict, dr_dict = extract_second_job_content(second_job_block)

    reasoning = ""
    not_used = ""

    print(used, " ", name)

    if "Default" in used or "default reasoning" in used.lower():
        reasoning = "DR"
        not_used = "CSR"
    elif "common sense" or "commonsense" in used.lower():
        reasoning = "CSR"
        not_used = "DR"

    result = {
        "mti": inference,
        "mti_why": why,
        "used_reasoning": reasoning,
        "not_used_reasoning": not_used,
        "explanation": explanation,
        "second_job_csr": csr_dict,
        "second_job_dr": dr_dict,
    }

    return result

def split_jobs_from_response(response_text):

    first_match = re.search(r'First Job\s*(.*?)\s*Second Job', response_text, re.DOTALL)
    first_block = first_match.group(1).strip() if first_match else ""

    second_match = re.search(r'Second Job\s*(.*)', response_text, re.DOTALL)
    second_block = second_match.group(1).strip() if second_match else ""

    return first_block, second_block


def clean_text(text):
    """Pulisce il testo rimuovendo asterischi e normalizzando gli invii a capo"""
    if not text:
        return text

    # Rimuovi asterischi
    text = text.replace('*', '')

    # Normalizza gli invii a capo (sostituisce multipli invii a capo con uno singolo)
    text = re.sub(r'\n+', '\n', text)

    # Rimuovi invii a capo all'inizio e alla fine
    text = text.strip()
    return text


def extract_first_job_content(first_job_block) -> (str, str, str, str, str):
    inf_match = re.search(
        r"Most typical inference:\s*(.*?)\s*Why:\s*(.*?)(?:\s*Used reasoning:|$)",
        first_job_block,
        re.DOTALL
    )

    if inf_match:
        most_typical_inference = clean_text(inf_match.group(1).strip())
        why_reasoning = clean_text(inf_match.group(2).strip())
    else:
        most_typical_inference = ""
        why_reasoning = ""

    reasoning_match = re.search(
        r"Used reasoning:\s*(.*?)\s*Not used reasoning:\s*(.*)",
        first_job_block,
        re.DOTALL
    )

    if reasoning_match:
        used_reasoning = clean_text(reasoning_match.group(1).strip())
        not_used_reasoning = clean_text(reasoning_match.group(2).strip())
    else:
        used_reasoning = ""
        not_used_reasoning = ""

    explanation_match = re.search(
        r"Explanation:\s*(.*?)$",
        first_job_block,
        re.DOTALL
    )

    if explanation_match:
        explanation = clean_text(explanation_match.group(1).strip())
    else:
        explanation = ""

    return most_typical_inference, why_reasoning, used_reasoning, not_used_reasoning, explanation


def extract_first_job_content_only_used(first_job_block) -> (str, str, str, str):

    inf_match = re.search(
        r"Most typical inference:\s*(.*?)\s*Why:\s*(.*?)(?:\s*Used reasoning:|$)",
        first_job_block,
        re.DOTALL
    )

    if inf_match:
        most_typical_inference = clean_text(inf_match.group(1).strip())
        why_reasoning = clean_text(inf_match.group(2).strip())
    else:
        most_typical_inference = ""
        why_reasoning = ""

    reasoning_match = re.search(
        r"Used reasoning:\s*(.*?)\n",
        first_job_block,
        re.DOTALL
    )

    if reasoning_match:
        used_reasoning = clean_text(reasoning_match.group(1).strip())
    else:
        used_reasoning = ""

    explanation_match = re.search(
        r"Explanation:\s*(.*?)$",
        first_job_block,
        re.DOTALL
    )

    if explanation_match:
        explanation = clean_text(explanation_match.group(1).strip())
    else:
        explanation = ""

    return most_typical_inference, why_reasoning, used_reasoning, explanation


def extract_second_job_content(testo):
    parts = re.split(r'Default reasoning', testo, flags=re.IGNORECASE)

    common_text = parts[0].replace('Common sense', '').strip()
    common_sense = extract_inferences(common_text)

    default_text = parts[1].strip() if len(parts) > 1 else ""
    default_reasoning = extract_inferences(default_text)

    return common_sense, default_reasoning


def extract_inferences(testo):
    sentence_match = re.search(r'Sentence:\s*(.+)', testo)
    sentence = clean_text(sentence_match.group(1).strip()) if sentence_match else ""

    inferences = {}
    inferences_section = re.search(r'Inferences:\s*(.+?)(?=Reasoning:|$)', testo, re.DOTALL)

    if inferences_section:
        inferences_text = inferences_section.group(1)
        matches = re.findall(r'^\s*([^:\n]+):\s*(.+)', inferences_text, re.MULTILINE)

        for key, value in matches:
            inferences[clean_text(key.strip())] = clean_text(value.strip())

    return {
        'sentence': sentence,
        'inferences': inferences
    }

def clean_value(text):
    """Rimuove la formattazione Markdown (**) e i newline, quindi pulisce gli spazi."""
    # Rimuove '**' e '\n'
    cleaned_text = re.sub(r'[\*\n]', '', text)
    # Rimuove spazi bianchi all'inizio e alla fine
    return cleaned_text.strip()

def extract_changes(testo):
    regex_full = r"Decision:\s*(.*?)\s*New choice(?:\s*\(if changed\))?:\s*(.*?)\s*Type of reasoning(?:\s*\(if changed\))?:\s*(.*?)\s*Explanation:\s*(.*)"

    match = re.search(regex_full, testo, re.DOTALL)

    decision = ""
    new_choice = ""
    type_of_reasoning = ""
    explanation = ""

    if match:
        raw_decision = match.group(1)
        raw_new_choice = match.group(2)
        raw_type_of_reasoning = match.group(3)
        raw_explanation = match.group(4)

        # 2. Applica la pulizia a ciascun valore estratto
        decision = clean_value(raw_decision)
        new_choice = clean_value(raw_new_choice)
        type_of_reasoning = clean_value(raw_type_of_reasoning)
        explanation = clean_value(raw_explanation)

    return decision, new_choice, type_of_reasoning, explanation

def new_entry_creator(index, gpt_result, gemini_result, claude_result):

    first_job_results = [
        index,
        gpt_result["mti"], gemini_result["mti"], claude_result["mti"],
        gpt_result["used_reasoning"], gemini_result["used_reasoning"], claude_result["used_reasoning"],
        gpt_result["explanation"], gemini_result["explanation"], claude_result["explanation"]
    ]

    second_job_results = [{} for i in range(11)]

    overall_result = {
        "gpt": gpt_result,
        "gemini": gemini_result,
        "claude": claude_result,
    }

    for llm_name, llm_result in overall_result.items():
        second_job_results[0][llm_name] = llm_result["second_job_csr"]["sentence"]
        second_job_results[1][llm_name] = llm_result["second_job_dr"]["sentence"]

        count = 2
        for key, value in llm_result["second_job_csr"]["inferences"].items():
            second_job_results[count][f"{llm_name}_csr"] = value
            count += 1

        count = 2
        for key, value in llm_result["second_job_dr"]["inferences"].items():
            second_job_results[count][f"{llm_name}_dr"] = value
            count += 1

    return first_job_results + second_job_results