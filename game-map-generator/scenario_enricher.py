from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "google/flan-t5-large"  # ✅ Yeni model: daha akıllı, CPU'da sorunsuz

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def enrich_scenario_local(user_input):
    prompt = f"""
Analyze the following game scenario. Return a detailed map design plan including theme, environment, at least 3 locations, 2 enemy types, core map components, and a 2-sentence story expansion.

Scenario: {user_input}

Format:
- Map Theme:
- Environment Type:
- Key Locations (3+):
- Enemies/Obstacles (2+):
- Map Components:
- Expanded Narrative (2–3 sentences):
Respond in English.
"""

    response = generator(prompt, max_new_tokens=300, temperature=0.9)
    return response[0]["generated_text"]
