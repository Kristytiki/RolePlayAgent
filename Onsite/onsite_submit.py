"""
Single-shot Chaiverse submitter — kept as the original onsite reference.

For the portfolio submitter (multiple model_repos, formatters, gen-param sweeps,
full ledger to submissions.{json,xlsx}) use Onsite/submit_batch.py instead.

A submission takes ~10-15 minutes to spin up and ~90 minutes to accumulate
~5k-10k preference battles. Track each submission_id; use the Chaiverse SDK
(see ~/.claude/skills/chaiverse-winrate/refresh.py) to pull win-rates.
"""
import os
import sys

import requests


# Read credentials from environment. Never commit literal values.
CHAI_DEVELOPER_KEY = os.environ.get("CHAI_DEVELOPER_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not CHAI_DEVELOPER_KEY or not HF_TOKEN:
    sys.exit("set CHAI_DEVELOPER_KEY and HF_TOKEN env vars before running")

def submit_model(model_submission: dict, developer_key: str):
    url = "http://guanaco-submitter-v2.guanaco-backend.kchai-google-us-east4.chaiverse.com/models/submit"
    headers = {"Authorization": f"Bearer {developer_key}"}
    response = requests.post(url, headers=headers, json=model_submission)
    assert response.status_code == 200, response.text
    return response.json()


def main():
    #model_name = 'Qwen/Qwen2.5-3B-Instruct'
    model_name = 'google/gemma-4-E2B-it'
    model_submission = {
        'model_repo': model_name,
        'generation_params': {
            'temperature': 1.0,
            'top_p': 1.0,
            'min_p': 0.0,
            'top_k': 40,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'stopping_words': ['\n'],
            'max_input_tokens': 2048,
            'best_of': 8,
            'max_output_tokens': 64,
		},
        'hf_token': HF_TOKEN,
        'formatter': {
            'memory_template': '<|im_start|>system\n{memory}<|im_end|>\n',
            'prompt_template': '<|im_start|>user\n{prompt}<|im_end|>\n',
            'bot_template': '<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n',
            'user_template': '<|im_start|>user\n{user_name}: {message}<|im_end|>\n',
            'response_template': '<|im_start|>assistant\n{bot_name}:',
            'truncate_by_message': True,
        },
        'platform': 'vllm',
    }
    model = submit_model(model_submission, developer_key=CHAI_DEVELOPER_KEY)
    print(f'Submission ID: {model}')
    sid = model.get("submission_id") if isinstance(model, dict) else None
    if sid:
        print(f"URL: https://console.chaiverse.com/models/{sid}")
    else:
        print("WARNING: response did not contain submission_id; raw payload above")

if __name__ == '__main__':
    main()
