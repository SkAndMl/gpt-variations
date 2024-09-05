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


def evaluate_commonsenseqa():
    
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

def evaluate_anli():
    anli = load_dataset('anli', split='dev_r1')  
    labels = ['entailment', 'contradiction', 'neutral']
    
    for model_id in models:
        num_correct, num_total = 0, 0
        model = models[model_id]
        model.eval()
        
        for example in anli:
            premise = example['premise']
            hypothesis = example['hypothesis']
            correct_label = example['label']
            sequences = [
                f"Premise: {premise}. Hypothesis: {hypothesis}. Premise entails Hypothesis",
                f"Premise: {premise}. Hypothesis: {hypothesis}. Premise contradicts Hypothesis",
                f"Premise: {premise}. Hypothesis: {hypothesis}. Premise neutral Hypothesis."
            ]

            label_probs = {}
            for label, sequence in zip(labels, sequences):
                input_ids = tokenizer.encode(sequence, return_tensors='pt')
                with torch.inference_mode():
                    logits, _ = model(input_ids)
                    sequence_prob = torch.sum(torch.log_softmax(logits, dim=-1)).item()
                    label_probs[label] = sequence_prob

            predicted_label = max(label_probs, key=label_probs.get)
            num_correct += 1 if predicted_label==labels[correct_label] else 0
            num_total += 1
        
        with open("logs.txt", "a") as f:
            f.write(f'anli, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


def evaluate_piqa():

    piqa = load_dataset('piqa', split='validation')  
    for model_id in models:
        model = models[model_id]
        model.eval()

        num_correct, num_total = 0, 0
        for example in piqa:
            question = example['goal']
            choices = [example['sol1'], example['sol2']]
            correct_answer = choices[int(example['label'])] 
            option_probs = {}
            
            for option in choices:
                input_text = f"Question: {question} Answer: {option}"
                input_ids = tokenizer.encode(input_text, return_tensors='pt')

                with torch.inference_mode():
                    logits, _ = model(input_ids)
                    sequence_prob = torch.sum(torch.log_softmax(logits, dim=-1)).item()  
                    option_probs[option] = sequence_prob

            predicted_answer = max(option_probs, key=option_probs.get)
            num_correct += 1 if predicted_answer==correct_answer else 0
            num_total += 1

        with open("logs.txt", "a") as f:
            f.write(f'piqa, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


def evaluate_copa():

    copa = load_dataset('super_glue', 'copa', split='validation')  
    for model_id in models:
        model = models[model_id]
        model.eval()
        num_correct, num_total = 0, 0

        def calculate_probability(sequence):
            input_ids = tokenizer.encode(sequence, return_tensors='pt')
            with torch.no_grad():
                logits, _ = model(input_ids)
                return torch.sum(torch.log_softmax(logits, dim=-1)).item()  # Sum log probabilities
        
        for example in copa:
            premise = example['premise']
            choice1 = example['choice1']
            choice2 = example['choice2']
            correct_label = example['label']  # 0 or 1, indicating which choice is correct
    
            sequence1 = f"Premise: {premise} Choice: {choice1}"
            sequence2 = f"Premise: {premise} Choice: {choice2}"
    
            prob1 = calculate_probability(sequence1)
            prob2 = calculate_probability(sequence2)

            predicted_label = 0 if prob1 > prob2 else 1
            num_correct += 1 if predicted_label==correct_label else 0
            num_total += 1
        
        with open("logs.txt", "a") as f:
            f.write(f'copa, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


def evaluate_arc_easy():
    
    arc_easy = load_dataset('ai2_arc', 'ARC-Easy', split='validation')  # Limiting to the first 10 examples
    for model_id in models:
        model = models[model_id]
        model.eval()
        num_correct, num_total = 0, 0
        for example in arc_easy:
            question = example['question']
            choices = example['choices']['text']
            correct_label = example['answerKey']  # 'A', 'B', etc.
            if ord(correct_label)>=65: correct_answer = choices[ord(correct_label) - ord('A')]  
            else: correct_answer = choices[int(correct_label)-1]
            option_probs = {}
            
            for option in choices:
                input_text = f"Question: {question} Answer: {option}"
                input_ids = tokenizer.encode(input_text, return_tensors='pt')

                with torch.inference_mode():
                    logits, _ = model(input_ids)
                    sequence_prob = torch.sum(torch.log_softmax(logits, dim=-1)).item()  # Sum log probabilities
                    option_probs[option] = sequence_prob

            predicted_answer = max(option_probs, key=option_probs.get)
            num_correct += 1 if predicted_answer==correct_answer else 0
            num_total += 1

        with open("logs.txt", "a") as f:
            f.write(f'arc_easy, {model_id}: {num_correct}/{num_total} -> {num_correct/num_total}\n')


if __name__ == "__main__":
    evaluate_arc_easy()