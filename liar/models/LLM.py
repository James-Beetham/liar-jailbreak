# From https://github.com/facebookresearch/advprompter

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from contextlib import nullcontext
import asyncio,time,typing

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessor

from liar.models.LLM_sequence import Seq, MergedSeq, msg_to_seq
from liar.models.LLM_utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)
import liar.conf.model.base_llm as base_llm_cfg

class _ClampInfNanLogits(LogitsProcessor):
    def __init__(self, safe_id: int): self.safe_id = int(safe_id)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Replace NaN/±Inf with -inf
        scores = torch.nan_to_num(scores, nan=float('-inf'), posinf=float('-inf'), neginf=float('-inf')) # type: ignore
        # If a row is all -inf, allow one safe token (e.g., EOS)
        all_neg_inf = torch.isneginf(scores).all(dim=-1)
        if all_neg_inf.any():
            scores[all_neg_inf] = float('-inf')
            scores[all_neg_inf, self.safe_id] = 0.0
        return scores

class LLM(nn.Module):
    def __init__(self, params:base_llm_cfg.BaseLLM, verbose=False) -> None:
        super().__init__()
        self.params = params
        self.verbose = verbose

        if len(self.params.llm_params.api)>0:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
            self.device = torch.device('cpu')
        if self.params.llm_params.api == 'together':
            import together
            self.client = together.AsyncTogether(api_key=self.params.llm_params.api_key)
        elif self.params.llm_params.api == 'openai':
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.params.llm_params.api_key)

        elif len(self.params.llm_params.api)>0:
            raise NotImplementedError(f'Unknokwn api: {self.params.llm_params.api}')
        else:
            self.model, self.tokenizer, self.embedding_matrix = llm_loader(
                llm_params=params.llm_params, verbose=verbose
            )

            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # TODO: This is a hack I added because Falcon-7b-isntruct doe snot have a pad token
                    # We might run into trouble here because the Seq class will automatically treat any eos_token as a pad_token and set the padding mask to 0 for this token
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            self.device = self.params.llm_params.device
            if self.params.allow_non_ascii:
                self.disallowed_ids = None
            else:
                self.disallowed_ids = get_nonascii_toks(self.tokenizer, device=self.device)

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, save_embedding_layers=True)

    def model_forward(self, query_seq:Seq, use_basemodel=False):
        # reorder such that all masked tokens are on the left
        mask = query_seq.mask
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

        with typing.cast(typing.Any,self.model).disable_adapter() \
                if use_basemodel and hasattr(self.model,'disable_adapter') else nullcontext(): 
            if query_seq.is_hard:
                ids = query_seq.ids
                sorted_ids = ids.gather(1, indices)
                shifted_sorted_pred_logits = self.model(
                    input_ids=sorted_ids, attention_mask=sorted_mask
                ).logits
            else:
                ids,sorted_ids = [],[]
                embeds = query_seq.get_embed(self.embedding_matrix)
                indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
                sorted_embeds = embeds.gather(1, indices_extended)
                shifted_sorted_pred_logits = self.model(
                    inputs_embeds=sorted_embeds, attention_mask=sorted_mask
                ).logits

        # reverse the sort to get the original order (also account for the shift)
        dummy_pred_logits = torch.zeros_like(shifted_sorted_pred_logits[:, :1, :])
        sorted_pred_logits = torch.cat(
            [dummy_pred_logits, shifted_sorted_pred_logits[:, :-1, :]], dim=1
        )
        reverse_indices = indices.argsort(dim=1)
        reverse_indices_extended = reverse_indices[:, :, None].repeat(
            1, 1, sorted_pred_logits.shape[-1]
        )
        shifted_pred_logits = sorted_pred_logits.gather(1, reverse_indices_extended)
        pred_logits = torch.cat(
            [shifted_pred_logits[:, 1:, :], shifted_sorted_pred_logits[:, -1:, :]],
            dim=1,
        )

        if self.disallowed_ids is not None:
            pred_logits[:, :, self.disallowed_ids] = -1e4 if pred_logits.dtype == torch.float16 else -1e10
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            for i in range(pred_logits.shape[0]):
                if torch.isnan(pred_logits[i]).any():
                    print(i, "-th logits..........", pred_logits[i])
                    print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
                    if len(ids)>i: print("ids........", ids[i])
                    print("sorted_masks.......", sorted_mask[i])
                    if len(sorted_ids)>i: print("sorted_ids", sorted_ids[i])
            raise RuntimeError(f"NaN in pred_logits: {pred_logits}")
        new_mask = torch.ones_like(mask)
        new_mask[:, :-1] = mask[:, 1:]
        seq = Seq(
            logits=pred_logits,
            mask=new_mask,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return seq

    @autocast_decorator
    def compute_pred_loss_teacher_forced(self, loss_params, label=None, **kwargs):
        gen_seqs:typing.Any = self.generate_teacher_forced(**kwargs)
        if label is None:
            label = gen_seqs.response_teacher
        loss_return:typing.Any = loss_seqs(gen_seqs.response_dist, label, **loss_params)

        pred_loss_return = ReturnStruct(
            loss=loss_return.loss,
            loss_masked=loss_return.loss_masked,
            loss_batch=loss_return.loss_batch,
            query=gen_seqs.query,
            response_teacher=gen_seqs.response_teacher,
            response_dist=gen_seqs.response_dist,
            label=label,
            perplexity=gen_seqs.perplexity,
            perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        )
        return pred_loss_return

    @autocast_decorator
    def generate_teacher_forced(
        self, key, detach_query=False, use_basemodel=False, **context
    ):
        query_seq, response_teacher_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        ) # type: ignore
        if response_teacher_seq is None:
            raise ValueError(f"Key {key} not found in prompt template.")
        assert not response_teacher_seq.is_empty
        full_query_seq = MergedSeq([query_seq, response_teacher_seq]) # type: ignore
        if detach_query:
            full_query_seq = full_query_seq.clone().detach()

        pred_full_query_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel # type: ignore
        ) # type: ignore
        response_dist_seq = pred_full_query_seq[
            :, -response_teacher_seq.seq_len - 1 : -1
        ]
        perplexity, perplexity_per_token_masked = compute_perplexity(
            id_seq=response_teacher_seq, likelihood_seq=response_dist_seq
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_teacher=response_teacher_seq,
            response_dist=response_dist_seq,
            perplexity=perplexity,
            perplexity_per_token_masked=perplexity_per_token_masked,
        )
        return return_seqs

    def get_next_token(self, key, use_basemodel=False, **context):
        query_seq, key_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        ) # type: ignore
        full_query_seq = MergedSeq([query_seq, key_seq]) # type: ignore

        pred_dist_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel # type: ignore
        ) # type: ignore
        next_dist_seq = pred_dist_seq[:, -1:]

        return_seqs = ReturnStruct(query=full_query_seq, response_dist=next_dist_seq)
        return return_seqs

    def generate_autoregressive(
        self, key, use_basemodel=False, max_new_tokens=None, **context
    ):
        if len(self.params.llm_params.api)>0:
            msgs = context['full_instruct']
            if type(msgs) is str: msgs = [msgs]
            elif type(msgs) is Seq: msgs = msgs.text
            ret = asyncio.run(self.async_chat_completion(msgs,max_new_tokens=max_new_tokens,
                                                         api=self.params.llm_params.api))
            return_seqs = ReturnStruct(
                query=Seq(tokenizer=self.tokenizer,device=self.device,text=msgs),
                response_sample=Seq(tokenizer=self.tokenizer,device=self.device,text=ret),
            )
            return return_seqs

        query_seq = typing.cast(MergedSeq, self.prepare_prompt(context, up_to_key=key))

        mask = query_seq.mask
        ids = query_seq.ids
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)
        sorted_ids = ids.gather(1, indices)

        generation_config = self.model.generation_config
        if self.disallowed_ids is not None:
            generation_config.suppress_tokens = self.disallowed_ids.tolist() # type: ignore
        generation_config.renormalize_logits = True # type: ignore

        if max_new_tokens is None:
            max_new_tokens = self.params.gen_params.max_new_tokens

        gen_params = dict(self.params.gen_params) # type: ignore
        gen_params["max_new_tokens"] = max_new_tokens # type: ignore

        safe_id = self.tokenizer.eos_token_id or (self.tokenizer.pad_token_id or 0)
        logits_processor = [_ClampInfNanLogits(safe_id)]


        with typing.cast(typing.Any,self.model).disable_adapter() if use_basemodel else nullcontext():
            output = self.model.generate(
                input_ids=sorted_ids,
                attention_mask=sorted_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
                **gen_params, # type: ignore
            )

        output_ids = output.sequences[:, ids.shape[1] :]

        response_sample_seq = Seq(
            ids=output_ids, tokenizer=self.tokenizer, device=self.device
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_sample=response_sample_seq,
        )
        return return_seqs

    def prepare_prompt(self, context, up_to_key=None, return_key_seq=False):
        seqs:list[Seq] = []
        seq = None
        for msg_dct in self.params.prompt_manager.prompt_template:
            if (
                up_to_key is not None
                and up_to_key == msg_dct.key
                and not return_key_seq
            ):
                break
            seq = msg_to_seq(
                msg=msg_dct.msg,
                tokenizer=self.tokenizer,
                device=self.device,
                context=context,
            )
            if up_to_key is not None and up_to_key == msg_dct.key and return_key_seq:
                break
            seqs.append(seq)

        merged_prompt_seq = MergedSeq(seqs)
        if return_key_seq:
            return merged_prompt_seq, seq
        else:
            return merged_prompt_seq

    async def async_chat_completion(self,messages:list[str],max_new_tokens,api:str):
        if self.params.prompt_manager.xq_wrapper:
            messages = [self.params.prompt_manager.xq_wrapper.format(v) for v in messages]
        delay = 60 # seconds
        res:list[str] = []
        for i in range(100):
            try:
                if api == 'together':
                    tasks = [
                        self.client.chat.completions.create(
                            model=self.params.llm_params.model_name,
                            messages=[{"role": "user", "content": message}],
                            # max_new_tokens=max_new_tokens,
                            max_tokens=max_new_tokens,
                        )
                        for message in messages
                    ]
                    responses = await asyncio.gather(*tasks)
                    res:list[str] = [v.choices[0].message.content for v in responses] # type: ignore
                elif api == 'openai':
                    tasks = [
                        self.client.responses.create( # type: ignore
                            model=self.params.llm_params.model_name,
                            input=message,
                            max_output_tokens=max_new_tokens
                        )
                        for message in messages
                    ]
                    responses = await asyncio.gather(*tasks)
                    res:list[str] = [v.output_text for v in responses]
                else: raise NotImplementedError(f'Unknown api: {api}')
            except Exception as e:
                if i == 2: raise e
                print(f'Exception, delaying {delay}s:')
                print(e)
                time.sleep(delay)
                delay *= 2
                delay = min(60*10,delay) # 10min max delay

        return res
