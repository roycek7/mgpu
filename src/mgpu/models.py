from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SamplingParams(BaseModel):
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    max_tokens: int = 256
    min_tokens: int = 1
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: Optional[bool] = True
    skip_special_tokens: bool = False
    spaces_between_special_tokens: bool = True
    logits_processors: Optional[List] = None
    truncate_prompt_tokens: Optional[int] = None
    guided_decoding: Optional[Dict] = None
    logit_bias: Optional[Dict[int, float]] = None
    allowed_token_ids: Optional[List[int]] = None


class LLMParams(BaseModel):
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: Optional[int] = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4.0
    cpu_offload_gb: float = 0.0
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    hf_overrides: Optional[Dict[str, Any]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    task: str = "auto"
    override_pooler_config: Optional[Dict[str, Any]] = None
