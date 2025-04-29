import openai
from src.utils.constants import *
import clip
import json
import re
from openai import OpenAI
import torch
from unsloth import FastLanguageModel

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Model cache for Unsloth models
MODEL_CACHE = {}

def get_model(engine):
    """Get or load a model from cache.
    
    Args:
        engine: The model name, e.g. "unsloth/Llama-3.2-3B-Instruct"
        
    Returns:
        Tuple of (model, tokenizer) if Unsloth model, None otherwise
    """
    if engine.startswith("unsloth/"):
        if engine not in MODEL_CACHE:
            print(f"Loading Unsloth model: {engine}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=engine,  # Pass the full model name
                max_seq_length=2048,
                dtype=torch.float16,
            )
            MODEL_CACHE[engine] = (model, tokenizer)
        return MODEL_CACHE[engine]
    return None

#@title LLM Cache
overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}
ENGINE = "gpt-4o-mini"  # Updated to GPT-4o. Use "gpt-4o" if you have access.
#@title LLM Scoring
def gpt3_call(engine=ENGINE, prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
  #breakpoint()
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    #breakpoint()
    response = LLM_CACHE[id]
  else:
    #breakpoint()
    response = client.completions.create(model=engine,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
  return response

def _parse_scores(raw_string: str, n: int):
    """Utility to convert the assistant response into a list of floats.
    The assistant is instructed to output a JSON list, but we also try to
    handle other simple formats gracefully. Returns a list of *n* floats.
    """
    raw_string = raw_string.strip()
    # First try JSON list / dict
    try:
        parsed = json.loads(raw_string)
        if isinstance(parsed, list):
            scores = parsed
        elif isinstance(parsed, dict):
            # Assume dict of index->score or option->score
            scores = [parsed[k] for k in sorted(parsed.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
        else:
            raise ValueError
    except Exception:
        # Fallback: extract all floats
        floats = re.findall(r"[-+]?[0-9]*\.?[0-9]+", raw_string)
        scores = [float(x) for x in floats]
    # Pad / truncate to expected length
    if len(scores) < n:
        scores += [0.0] * (n - len(scores))
    return scores[:n]

def gpt3_scoring(query, options, engine=ENGINE, limit_num_options=None, verbose=False, print_tokens=False, scoring_method="sequence", **kwargs):
    """Return a probability-like score for each option using GPT-4o chat API and logprobs.
    
    This implementation has two methods for scoring:
    2. "direct": Ask the model directly for the next action number and extract the logprobs
       of the number tokens corresponding to the top 10 most likely tokens.
    """
    if limit_num_options:
        options = options[:limit_num_options]
    
    if scoring_method == "direct":
        # Direct ranking method: Ask the model which option number to pick next
        numbered_options = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        user_prompt = (
            f"Task description:\n{query}\n\n"
            f"Given the following options, respond with ONLY the NUMBER of the option "
            f"that should be executed next. Only choose 'done()' if the task is truly complete. "
            f"If there are still actions to take to complete the task, choose one of the robot actions instead:\n{numbered_options}"
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful robotics planning assistant. "
                    "You understand the semantics of the SayCan pick-and-place API and "
                    "can judge which low-level action best progresses a high-level goal. "
                    "The 'done()' command should ONLY be used when all requirements of the task "
                    "have been satisfied. If there are still actions that need to be taken, "
                    "you should choose an action that makes progress toward completing the task."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        
        if engine.startswith("unsloth/"):
            # For Unsloth models, get top 10 logits
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            model, tokenizer = get_model(engine)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Get logits for the next token
                
                # Get top 10 tokens and their logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                top_logprobs, top_indices = torch.topk(logprobs, k=10)
                top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
                
                # Debug print
                print("\nTop tokens and their logprobs:")
                for token, logprob in zip(top_tokens, top_logprobs[0]):
                    print(f"Token: '{token}', Logprob: {float(logprob):.4f}")
                
                # Initialize scores with very low values
                scores = {opt: -100.0 for opt in options}
                
                # Score based on number tokens in top 10
                for token, logprob in zip(top_tokens, top_logprobs[0]):
                    # Skip whitespace and punctuation
                    if not token.strip() or token.strip() in [' ', '\n', '\t', '(', ')', ',', '.', '#', '-']:
                        continue
                        
                    # Check for numbers
                    if re.match(r'^\d+$', token.strip()):
                        try:
                            option_num = int(token.strip())
                            if 1 <= option_num <= len(options):
                                option = options[option_num-1]
                                scores[option] = float(logprob)  # Use the logprob value
                                print(f"Found number {option_num}, scoring option: {option} with logprob: {float(logprob):.4f}")
                        except (ValueError, IndexError):
                            pass
                    # Check for number with period
                    elif re.match(r'^\d+\.', token.strip()):
                        try:
                            option_num = int(token.strip().replace('.', ''))
                            if 1 <= option_num <= len(options):
                                option = options[option_num-1]
                                scores[option] = float(logprob)  # Use the logprob value
                                print(f"Found number {option_num} with period, scoring option: {option} with logprob: {float(logprob):.4f}")
                        except (ValueError, IndexError):
                            pass
                
                # If no numbers found, try number words
                if all(score == -100.0 for score in scores.values()):
                    print("\nNo numbers found, trying number words...")
                    number_words = {
                        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
                    }
                    
                    for token, logprob in zip(top_tokens, top_logprobs[0]):
                        # Skip whitespace and punctuation
                        if not token.strip() or token.strip() in [' ', '\n', '\t', '(', ')', ',', '.', '#', '-']:
                            continue
                            
                        token = token.lower()
                        if token in number_words:
                            option_num = number_words[token]
                            if 1 <= option_num <= len(options):
                                option = options[option_num-1]
                                scores[option] = float(logprob)  # Use the logprob value
                                print(f"Found number word '{token}', scoring option: {option} with logprob: {float(logprob):.4f}")
                
                # If still no scores, try matching option text directly
                if all(score == -100.0 for score in scores.values()):
                    print("\nNo number words found, trying direct text matching...")
                    for token, logprob in zip(top_tokens, top_logprobs[0]):
                        # Skip whitespace and punctuation
                        if not token.strip() or token.strip() in [' ', '\n', '\t', '(', ')', ',', '.', '#', '-']:
                            continue
                            
                        token = token.strip()
                        for i, opt in enumerate(options):
                            # Only match if the token is a significant part of the option
                            if len(token) > 2 and token in opt:
                                scores[opt] = max(scores[opt], float(logprob))
                                print(f"Found matching token '{token}' in option: {opt}, scoring with logprob: {float(logprob):.4f}")
                
                # Convert logprobs to probabilities for better interpretation
                if any(score != -100.0 for score in scores.values()):
                    print("\nFinal scores before conversion:")
                    for opt, score in scores.items():
                        if score != -100.0:
                            print(f"Option: {opt}, Logprob score: {score:.4f}")
                    
                    # Normalize scores to sum to 1
                    total = sum(torch.exp(torch.tensor(score)) for score in scores.values() if score != -100.0)
                    if total > 0:
                        for opt in scores:
                            if scores[opt] != -100.0:
                                scores[opt] = torch.exp(torch.tensor(scores[opt])).item() / total
                            else:
                                scores[opt] = 0.0
                    
                    print("\nFinal scores after conversion to probabilities:")
                    for opt, score in scores.items():
                        if score > 0:
                            print(f"Option: {opt}, Probability score: {score:.4f}")
                else:
                    print("\nNo valid scores found for any options!")
                
                response = tokenizer.decode(top_indices[0][0])  # Get the top token as response
        else:
            # Original GPT-4 scoring pipeline remains unchanged
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0,
                max_tokens=10,
                logprobs=True,
                top_logprobs=10
            )
            
            # Extract the logprobs from the API response
            completion_tokens = response.choices[0].logprobs.content
            response_text = response.choices[0].message.content
            
            # Initialize scores with very low values
            scores = {opt: -100.0 for opt in options}
            
            # Evaluate all option numbers from top_logprobs
            for token_info in completion_tokens:
                # Look at all the top logprobs for this token position
                for logprob_info in token_info.top_logprobs:
                    token_text = logprob_info.token
                    # Check if this is a digit
                    if re.match(r'^\d+$', token_text.strip()):
                        try:
                            option_num = int(token_text.strip())
                            # If it's a valid option number
                            if 1 <= option_num <= len(options):
                                # Get the corresponding option
                                option = options[option_num-1]
                                # Update score if this is better than current score
                                scores[option] = max(scores[option], logprob_info.logprob)
                        except (ValueError, IndexError):
                            pass  # Not a valid number
                    # Also check for completed tokens like "1." or "2."
                    elif re.match(r'^\d+\.', token_text.strip()):
                        try:
                            option_num = int(token_text.strip().replace('.', ''))
                            if 1 <= option_num <= len(options):
                                option = options[option_num-1]
                                scores[option] = max(scores[option], logprob_info.logprob)
                        except (ValueError, IndexError):
                            pass  # Not a valid number format
            
            # If none of the numbers had a score, look for words like "first", "second", etc.
            if all(score == -100.0 for score in scores.values()):
                number_words = {
                    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
                }
                
                for token_info in completion_tokens:
                    for logprob_info in token_info.top_logprobs:
                        token_text = logprob_info.token.lower()
                        if token_text in number_words:
                            option_num = number_words[token_text]
                            if 1 <= option_num <= len(options):
                                option = options[option_num-1]
                                scores[option] = max(scores[option], logprob_info.logprob)
            
            # As a last fallback, try to parse the full response for a number
            if all(score == -100.0 for score in scores.values()):
                try:
                    number_match = re.search(r'\d+', response_text)
                    if number_match:
                        selected_num = int(number_match.group(0))
                        if 1 <= selected_num <= len(options):
                            scores[options[selected_num-1]] = -1.0  # Better than -100 but still low
                except:
                    pass
    else:  # Default "sequence" method
        # Build a prompt that encourages the model to generate each option
        user_prompt = f"Task description:\n{query}\n\nI'll evaluate the following options one by one:\n"
        for i, opt in enumerate(options):
            user_prompt += f"{i+1}. {opt}\n"
        
        # Add a message to prime the model to start generating scores
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful robotics planning assistant. "
                    "You understand the semantics of the SayCan pick-and-place API and "
                    "can judge which low-level action best progresses a high-level goal."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        
        if engine.startswith("unsloth/"):
            # For Unsloth models, get top 10 logits
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            model, tokenizer = get_model(engine)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Get logits for the next token
                
                # Get top 10 tokens and their logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                top_logprobs, top_indices = torch.topk(logprobs, k=10)
                top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
                
                # Score each option by finding its tokens in the completion
                scores = {}
                for opt in options:
                    # Get a clean version of the option for matching
                    clean_opt = opt
                    if "robot.pick_and_place" in opt:
                        # Extract the objects from the API command
                        parts = opt.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
                        if len(parts) == 2:
                            pick_obj, place_obj = parts
                            clean_opt = f"{pick_obj} {place_obj}"
                    
                    # Score based on token logprobs
                    option_score = -100.0
                    for token, logprob in zip(top_tokens, top_logprobs[0]):
                        if token in clean_opt:
                            option_score = max(option_score, float(logprob))
                    scores[opt] = option_score
                
                response = tokenizer.decode(top_indices[0][0])  # Get the top token as response
        else:
            # Original GPT-4 scoring pipeline remains unchanged
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                logprobs=True,
                top_logprobs=5
            )
            
            # Extract the logprobs from the API response
            completion_tokens = response.choices[0].logprobs.content
            
            # Score each option by finding its tokens in the completion and summing logprobs
            scores = {}
            for opt in options:
                # Get a clean version of the option for matching
                clean_opt = opt
                if "robot.pick_and_place" in opt:
                    # Extract the objects from the API command
                    parts = opt.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
                    if len(parts) == 2:
                        pick_obj, place_obj = parts
                        clean_opt = f"{pick_obj} {place_obj}"
                
                # Find all tokens that might correspond to this option
                opt_tokens = []
                opt_logprobs = []
                
                # Look for sequences in the completion that match parts of the option
                for token_info in completion_tokens:
                    # Check if this token could be part of our option
                    token_text = token_info.token
                    
                    # If the token is part of the option text, add its logprob
                    if token_text in clean_opt:
                        opt_tokens.append(token_text)
                        opt_logprobs.append(token_info.logprob)
                
                # Calculate score based on logprobs of relevant tokens
                if opt_logprobs:
                    # Use average logprob as the score
                    scores[opt] = sum(opt_logprobs) / len(opt_logprobs)
                else:
                    # No matching tokens found, assign a very low score
                    scores[opt] = -100.0
        
        # Ensure all options have a score
        for opt in options:
            if opt not in scores:
                scores[opt] = -100.0
    
    if verbose:
        print(f"Scored {len(options)} options with {scoring_method} method")
        sorted_options = sorted(options, key=lambda opt: scores[opt], reverse=True)
        for opt in sorted_options:
            print(f"{scores[opt]:.3f}\t{opt}")
    
    if print_tokens and not engine.startswith("unsloth/"):
        try:
            print(f"Tokens used: {response.usage.total_tokens}")
        except:
            print("Could not print token usage")

    return scores, response

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options


#@markdown Load CLIP model.

# torch.cuda.set_per_process_memory_fraction(0.9, None)
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", clip_model.visual.input_resolution)
print("Context length:", clip_model.context_length)
print("Vocab size:", clip_model.vocab_size)
def encode_text(text):
  return clip_model.encode_text(text)
