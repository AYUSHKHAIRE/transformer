"""
DRAFT
"""

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt, set tensor to batch size of 1
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions for the next token as the last element of the output
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Stop if we predict the end token
            if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())

# Test the model with some prompts
test_prompts = [
    "Once upon a time,",
    "We the people of the",
    "In the beginning was the",
]

print("\nGenerating sample texts:")
for prompt in test_prompts:
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 80)