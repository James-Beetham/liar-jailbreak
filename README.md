# LIAR Jailbreak

This repository contains the reference implementation for the paper:

[Jailbreaks as Inference-Time Alignment: A Framework for Understanding Safety Failures in LLMs](https://arxiv.org/abs/2412.05232)


## Setup

Create conda environment (or venv):
`conda create -n liar python=3.12 -y`

Activate environment, install `torch` and `torchvision` (instructions [here](https://pytorch.org/get-started/locally/)).

Install requirements:
`pip install -r requirements.txt`

Run either `scripts/run_liar.sh` or through VSCode Debugger `.vscode/launch.json`.

Results are generated in `outputs/{run_name}` (hydra generates logs in `outputs/{date}/...`).
`debug.txt` shows basic x,q,y set, dump.json shows outputs for each iteration,
and `results.txt` provide summary jailbreak results.

**Note:**
- Some warnings of weight mis-match (especially with GPT2) can be ignored if the outputs look fine (e.g. natural).
- Some warnings regarding generation params are fine (e.g. `max_new_tokens`, etc).


## Attribution

This project includes code adapted from:

AdvPrompter  
© Meta Platforms, Inc.  
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

Original repository: https://github.com/facebookresearch/advprompter  
License: https://creativecommons.org/licenses/by-nc/4.0/


## Citation

If you use this code or find it helpful in your research, please cite:

```
@misc{beetham2025liarjailbreaksasinference,
  title={Jailbreaks as Inference-Time Alignment: A Framework for Understanding Safety Failures in LLMs}, 
  author={James Beetham and Souradip Chakraborty and Mengdi Wang and Furong Huang and Amrit Singh Bedi and Mubarak Shah},
  year={2025},
  eprint={2412.05232},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.05232}, 
}
```
