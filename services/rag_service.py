from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

llm = None


def get_model():
    global llm

    if llm is None:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )

    return llm


def generate_answer(question: str):

    model = get_model()

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer