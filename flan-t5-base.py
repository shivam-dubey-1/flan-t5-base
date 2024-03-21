from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class FlanT5Base:
    @staticmethod
    def generate_response(prompt, max_tokens=50) -> str:
        # Tokenize prompt
        inputs = tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)

        # Generate response
        output = model.generate(
            inputs["input_ids"],
            min_length = 2,
            max_new_tokens = 200,
            length_penalty = 1.4,
            num_beams = 12,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            top_k=150,
            top_p=0.92,
            repetition_penalty = 2.1,
            temperature=0.2
        )

        # Decode and return response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

#@serve.deployment()
class FlanT5(FlanT5Base):
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    prompt = "what is milky way?"
    flan = FlanT5()
    print(flan.generate_response(prompt))
