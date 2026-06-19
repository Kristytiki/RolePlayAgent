# Chaiverse Submissions Log

Each entry: slug, submission_id, URL, full generation_params + formatter.
Win-rate / preferences are filled in manually after ~90min eval.

## `qwen25_3b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen2-5-3b-instruct_v19`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v19
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_1p5b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen2-5-1-5b-instruct_v1`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-1-5b-instruct_v1
- **model_repo**: `Qwen/Qwen2.5-1.5B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_0p5b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen2-5-0-5b-instruct_v3`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-0-5b-instruct_v3
- **model_repo**: `Qwen/Qwen2.5-0.5B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_1p7b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen3-1-7b_v1`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-1-7b_v1
- **model_repo**: `Qwen/Qwen3-1.7B`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_4b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen3-4b-instruct-2507_v10`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-4b-instruct-2507_v10
- **model_repo**: `Qwen/Qwen3-4B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_30b_a3b`  —  2026-06-19 18:22

- **submission_id**: `qwen-qwen3-30b-a3b-inst_16638_v3`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v3
- **model_repo**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `llama32_3b`  —  2026-06-19 18:22

- **submission_id**: `meta-llama-llama-3-2-3b_30223_v2`
- **url**: https://console.chaiverse.com/models/meta-llama-llama-3-2-3b_30223_v2
- **model_repo**: `meta-llama/Llama-3.2-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `llama32_1b`  —  2026-06-19 18:22

- **submission_id**: `meta-llama-llama-3-2-1b_53640_v2`
- **url**: https://console.chaiverse.com/models/meta-llama-llama-3-2-1b_53640_v2
- **model_repo**: `meta-llama/Llama-3.2-1B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma4_e2b`  —  2026-06-19 18:22

- **submission_id**: `google-gemma-4-e2b-it_v1`
- **url**: https://console.chaiverse.com/models/google-gemma-4-e2b-it_v1
- **model_repo**: `google/gemma-4-E2B-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma3_1b`  —  2026-06-19 18:22

- **submission_id**: `google-gemma-3-1b-it_v2`
- **url**: https://console.chaiverse.com/models/google-gemma-3-1b-it_v2
- **model_repo**: `google/gemma-3-1b-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `smollm2_1p7b`  —  2026-06-19 18:22

- **submission_id**: `huggingfacetb-smollm2-1-_7501_v1`
- **url**: https://console.chaiverse.com/models/huggingfacetb-smollm2-1-_7501_v1
- **model_repo**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_bo16`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen2-5-3b-instruct_v20`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v20
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_long`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen2-5-3b-instruct_v21`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v21
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 128
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_freqpen`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen2-5-3b-instruct_v22`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v22
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.3,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_conservative`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen2-5-3b-instruct_v23`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v23
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 0.8,
  "top_p": 0.9,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_coser_guide`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen2-5-3b-instruct_v24`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v24
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_30b_a3b_bo16`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-30b-a3b-inst_16638_v4`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v4
- **model_repo**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_30b_a3b_long`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-30b-a3b-inst_16638_v5`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v5
- **model_repo**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 128
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_4b_bo16`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-4b-instruct-2507_v11`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-4b-instruct-2507_v11
- **model_repo**: `Qwen/Qwen3-4B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_4b_long`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-4b-instruct-2507_v12`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-4b-instruct-2507_v12
- **model_repo**: `Qwen/Qwen3-4B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 128
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma4_e2b_bo16`  —  2026-06-19 18:26

- **submission_id**: `google-gemma-4-e2b-it_v2`
- **url**: https://console.chaiverse.com/models/google-gemma-4-e2b-it_v2
- **model_repo**: `google/gemma-4-E2B-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma4_e2b_long`  —  2026-06-19 18:26

- **submission_id**: `google-gemma-4-e2b-it_v3`
- **url**: https://console.chaiverse.com/models/google-gemma-4-e2b-it_v3
- **model_repo**: `google/gemma-4-E2B-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 128
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_30b_a3b_coser_guide`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-30b-a3b-inst_16638_v6`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v6
- **model_repo**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_4b_coser_guide`  —  2026-06-19 18:26

- **submission_id**: `qwen-qwen3-4b-instruct-2507_v13`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-4b-instruct-2507_v13
- **model_repo**: `Qwen/Qwen3-4B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma4_e2b_coser_guide`  —  2026-06-19 18:26

- **submission_id**: `google-gemma-4-e2b-it_v4`
- **url**: https://console.chaiverse.com/models/google-gemma-4-e2b-it_v4
- **model_repo**: `google/gemma-4-E2B-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_smoke`  —  2026-06-19 20:42

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_13112_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_13112_v1
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-smoke`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_smoke_bo16`  —  2026-06-19 20:42

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_13112_v2`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_13112_v2
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-smoke`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_smoke_coser_guide`  —  2026-06-19 20:42

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_13112_v3`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_13112_v3
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-smoke`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen25_3b_long`  —  2026-06-19 20:43

- **submission_id**: `qwen-qwen2-5-3b-instruct_v25`
- **url**: https://console.chaiverse.com/models/qwen-qwen2-5-3b-instruct_v25
- **model_repo**: `Qwen/Qwen2.5-3B-Instruct`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 80
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_30b_a3b_long`  —  2026-06-19 20:43

- **submission_id**: `qwen-qwen3-30b-a3b-inst_16638_v7`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v7
- **model_repo**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 80
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3_4b_long`  —  2026-06-19 20:43

- **submission_id**: `qwen-qwen3-4b-instruct-2507_v14`
- **url**: https://console.chaiverse.com/models/qwen-qwen3-4b-instruct-2507_v14
- **model_repo**: `Qwen/Qwen3-4B-Instruct-2507`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 80
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `gemma4_e2b_long`  —  2026-06-19 20:43

- **submission_id**: `google-gemma-4-e2b-it_v5`
- **url**: https://console.chaiverse.com/models/google-gemma-4-e2b-it_v5
- **model_repo**: `google/gemma-4-E2B-it`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 80
}
```

**formatter**:
```json
{
  "memory_template": "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template": "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template": "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step200`  —  2026-06-19 21:17

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_30253_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_30253_v1
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step200`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step200_bo16`  —  2026-06-19 21:17

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_30253_v2`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_30253_v2
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step200`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step200_coser_guide`  —  2026-06-19 21:17

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_30253_v3`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_30253_v3
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step200`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step500`  —  2026-06-19 21:24

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_40142_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_40142_v1
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step500`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step500_bo16`  —  2026-06-19 21:24

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_40142_v2`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_40142_v2
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step500`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_coser_step500_coser_guide`  —  2026-06-19 21:24

- **submission_id**: `zheqiwu-qwen2-5-3b-cose_40142_v3`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_40142_v3
- **model_repo**: `ZheqiWu/Qwen2.5-3B-CoSER-step500`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}\n\nUse [your thought] for thoughts which others can't see. Use (your action) for actions which others can see.<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_pippa_r32`  —  2026-06-19 22:11

- **submission_id**: `zheqiwu-qwen2-5-3b-pippa-r32_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-pippa-r32_v1
- **model_repo**: `ZheqiWu/Qwen2.5-3B-PIPPA-r32`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_pippa_r32_bo16`  —  2026-06-19 22:11

- **submission_id**: `zheqiwu-qwen2-5-3b-pippa-r32_v2`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-pippa-r32_v2
- **model_repo**: `ZheqiWu/Qwen2.5-3B-PIPPA-r32`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_pippa_r16`  —  2026-06-19 22:29

- **submission_id**: `zheqiwu-qwen2-5-3b-pippa-r16_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-pippa-r16_v1
- **model_repo**: `ZheqiWu/Qwen2.5-3B-PIPPA-r16`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `qwen3b_pippa_r16_bo16`  —  2026-06-19 22:29

- **submission_id**: `zheqiwu-qwen2-5-3b-pippa-r16_v2`
- **url**: https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-pippa-r16_v2
- **model_repo**: `ZheqiWu/Qwen2.5-3B-PIPPA-r16`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 16,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template": "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template": "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `llama32_3b_anime_step30`  —  2026-06-19 22:40

- **submission_id**: `zheqiwu-llama-3-2-3b-an_23097_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-llama-3-2-3b-an_23097_v1
- **model_repo**: `ZheqiWu/Llama-3.2-3B-Anime-step30`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `llama32_3b_anime_step60`  —  2026-06-19 22:40

- **submission_id**: `zheqiwu-llama-3-2-3b-an_12275_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-llama-3-2-3b-an_12275_v1
- **model_repo**: `ZheqiWu/Llama-3.2-3B-Anime-step60`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": true
}
```

---

## `llama32_3b_anime_step100`  —  2026-06-19 22:40

- **submission_id**: `zheqiwu-llama-3-2-3b-an_40296_v1`
- **url**: https://console.chaiverse.com/models/zheqiwu-llama-3-2-3b-an_40296_v1
- **model_repo**: `ZheqiWu/Llama-3.2-3B-Anime-step100`
- **platform**: `vllm`
- **win_rate**: _TBD_

**generation_params**:
```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": [
    "\n"
  ],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

**formatter**:
```json
{
  "memory_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": true
}
```

---

