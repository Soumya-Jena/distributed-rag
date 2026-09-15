from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from src.config import (
    GENERATION_MODEL,
    MAX_NEW_TOKENS,
)


class LocalGenerator:

    def __init__(self, model_name=GENERATION_MODEL):

        self.last_input_token_count = 0
        self.model_name = model_name

        print(
            f"Loading generation model: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        )

        self.model.eval()

        print(
            f"Generation model loaded."
        )

    def generate(
        self,
        messages,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):

        return "".join(
            self.generate_stream(
                messages,
                max_new_tokens,
            )
        ).strip()

    def generate_stream(
        self,
        messages,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):

        inputs = (
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        input_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(input_device)
            for key, value
            in inputs.items()
        }

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        self.last_input_token_count = int(
            inputs["input_ids"].shape[-1]
        )

        generation_arguments = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": (
                self.tokenizer.eos_token_id
            ),
        }

        generation_errors = []

        def run_generation():
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **generation_arguments
                    )
            except BaseException as error:
                generation_errors.append(error)
                streamer.end()

        thread = Thread(
            target=run_generation,
            daemon=True,
        )

        thread.start()

        for text_fragment in streamer:
            yield text_fragment

        thread.join()

        if generation_errors:
            raise generation_errors[0]

    def generate_batch(
        self,
        message_batches,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        """Generate several deterministic answers in one model invocation."""
        if not message_batches:
            return []

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer.apply_chat_template(
                message_batches,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_width = int(inputs["input_ids"].shape[-1])
            input_device = next(self.model.parameters()).device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output_ids[:, input_width:]
            return [
                text.strip()
                for text in self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            ]
        finally:
            self.tokenizer.padding_side = previous_padding_side
