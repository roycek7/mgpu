import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel
from vllm import LLM
from vllm.sampling_params import (
    SamplingParams,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mgpu")


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


def validate_params(llm_model: Type, sampling_model: Type):
    """
    Decorator to validate and convert dictionary parameters into Pydantic models.

    Args:
        llm_model (Type): The Pydantic model class for LLM parameters.
        sampling_model (Type): The Pydantic model class for sampling parameters.

    Returns:
        Callable: The wrapped function.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(
            prompts: List[str],
            llm_config: Dict[str, Any],
            sampling_config: Dict[str, Any],
            *args,
            **kwargs,
        ):
            # Validate and convert llm_config
            if isinstance(llm_config, dict):
                llm_config = llm_model(**llm_config)
            elif not isinstance(llm_config, llm_model):
                raise ValueError(
                    f"llm_config must be a {llm_model} or a dictionary."
                )

            # Validate and convert sampling_config
            if isinstance(sampling_config, dict):
                sampling_config = sampling_model(**sampling_config)
            elif not isinstance(sampling_config, sampling_model):
                raise ValueError(
                    f"sampling_config must be a {sampling_model} or a dictionary."
                )

            # Call the original function with the validated models
            return func(prompts, llm_config, sampling_config, *args, **kwargs)

        return wrapper

    return decorator


def generate_on_single_gpu(
    gpu_id: int,
    prompts: List[str],
    llm_config: LLMParams,
    sampling_config: SamplingParams,
) -> Any:
    """
    Function to run inference on a single GPU.

    Args:
        gpu_id (int): The GPU ID to use for inference.
        prompts (List[str]): A list of input prompts for the model.
        llm_config (LLMParams): Configuration for the LLM model.
        sampling_config (SamplingParams): Parameters for sampling during inference.

    Returns:
        Any: The generated outputs from the model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        logger.info(
            f"Initializing model on GPU {gpu_id} with {len(prompts)} prompts."
        )
        llm = llm = LLM(**llm_config.model_dump())
        outputs = llm.generate(prompts, sampling_config, use_tqdm=False)
        logger.info(f"Successfully generated outputs on GPU {gpu_id}.")
        return outputs

    except Exception as e:
        logger.error(f"Error on GPU {gpu_id}: {e}")
        raise


def split_list(data: List[Any], num_splits: int) -> List[List[Any]]:
    """
    Split a list into approximately equal parts.

    Args:
        data (List[Any]): The list to split.
        num_splits (int): The number of splits.

    Returns:
        List[List[Any]]: A list of sublists, each containing a portion of the original data.
    """
    return [
        data[i * len(data) // num_splits : (i + 1) * len(data) // num_splits]
        for i in range(num_splits)
    ]


@validate_params(LLMParams, SamplingParams)
def mgpu_inference(
    prompts: List[str],
    llm_config: LLMParams,
    sampling_config: SamplingParams,
    num_gpus: int = 8,
) -> List[str]:
    """
    Interface for running multi-GPU inference with customizable parameters.

    Args:
        prompts (List[str]): A list of input prompts to generate responses for.
        llm_config (LLMParams): Parameters for configuring the LLM model.
        sampling_params (SamplingParams): Parameters for sampling during inference.
        num_gpus (int): Number of GPUs to use for inference (default: 8).

    Returns:
        List[str]: A list of generated responses corresponding to the input prompts.
    """
    logger.info(f"Starting inference with {num_gpus} GPUs.")

    # Split prompts among GPUs
    split_prompts = split_list(prompts, num_gpus)

    all_outputs = []

    # Run inference sequentially on each GPU
    for gpu_id, gpu_prompts in enumerate(split_prompts):
        try:
            # Run inference for the current GPU
            output = generate_on_single_gpu(
                gpu_id, gpu_prompts, llm_config, sampling_config
            )
            all_outputs.extend(output)
        except Exception as e:
            logger.error(f"Error on GPU {gpu_id}: {str(e)}")

    logger.info("Inference completed successfully.")

    # Extract the text from model outputs
    responses = [output.outputs[0].text for output in all_outputs]
    return responses
