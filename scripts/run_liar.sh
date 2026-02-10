#!/bin/bash

python -m liar.liar_jailbreak \
  output_dir=outputs/vicuna_chat-gpt2 \
  repeat=100 \
  "model@target=vicuna_chat" \
  target.gen_params.max_new_tokens=32 \
  target.batch_size=8 \
  adv_llm.gen_params.max_new_tokens=30 \
  adv_llm.gen_params.do_sample=true \
  data.just_test=true
