from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from models import GPT, GPTConfig, ParallelGPT, LinearGPT, ConvGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
models = {
    "gpt": GPT(GPTConfig(vocab_size=50304)),
    "pgpt": ParallelGPT(GPTConfig(vocab_size=50304)),
    "cgpt": ConvGPT(GPTConfig(vocab_size=50304)),
    "lgpt": LinearGPT(GPTConfig(vocab_size=50304))   
}

for model_id in models:
    models[model_id].load_state_dict(torch.load(f"checkpoints/{model_id}.pt", map_location=device)['model'])

def evaluate_winogrande():
    
    winogrande = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    
    for model_id in models:
        model = models[model_id]
        model.eval()
        num_correct, num_total = 0, 0
        for example in winogrande:
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = option1 if example['answer'] == '1' else option2

            sentence_option1 = sentence.replace("_", option1)
            sentence_option2 = sentence.replace("_", option2)
            
            input_ids1 = tokenizer.encode(sentence_option1, return_tensors='pt')
            input_ids2 = tokenizer.encode(sentence_option2, return_tensors='pt')

            with torch.inference_mode():
                logits1, _ = model(input_ids1)
                logits2, _ = model(input_ids2)

                # Sum log probabilities to get sequence likelihoods
                prob1 = torch.sum(torch.log_softmax(logits1, dim=-1)).item()
                prob2 = torch.sum(torch.log_softmax(logits2, dim=-1)).item()

            predicted = option1 if prob1 > prob2 else option2
            if predicted==answer: num_correct += 1
            num_total += 1

        with open("logs.txt", "a") as f:
            f.write(f'winogrande, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


def evaluate_commonsenseqa(question, choices, correct_answer):
    
    commonsenseqa = load_dataset('commonsense_qa', split='validation')  # Limiting to the first 10 examples
    
    for model_id in models:
        num_correct, num_total = 0, 0
        model = models[model_id]
        model.eval()
        for example in commonsenseqa:
            option_probs = {}
            question = example['question']
            choices = {chr(65+i): choice for i, choice in enumerate(example['choices']['text'])}
            correct_answer = example['answerKey']
            for option in choices:
                input_text = f"{question} Answer: {choices[option]}"
                input_ids = tokenizer.encode(input_text, return_tensors='pt')

                with torch.inference_mode():
                    logits, _ = model(input_ids)
                    sequence_prob = torch.sum(torch.log_softmax(logits, dim=-1)).item()
                    option_probs[option] = sequence_prob

            predicted_answer = max(option_probs, key=option_probs.get)
            num_correct += 1 if predicted_answer==correct_answer else 0
            num_total += 1
        
        with open("logs.txt", "a") as f:
            f.write(f'commonsenseqa, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


if __name__ == "__main__":
    evaluate_winogrande()
    evaluate_commonsenseqa()