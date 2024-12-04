import logging
import os
from typing import Any, List

from vllm import LLM
from vllm.sampling_params import (
    SamplingParams,
)

from .models import LLMParams
from .utils import split_into_chunks
from .validators import validate_and_convert_configs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mgpu")


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
    # Set the CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize and generate outputs
    try:
        logger.info(f"Initializing LLM on GPU {gpu_id} with {len(prompts)} prompts.")

        # Create the model instance using llm_config
        llm = LLM(**llm_config.model_dump())

        logger.info(
            f"Model initialized successfully on GPU {gpu_id}. Starting generation."
        )

        # Generate outputs
        outputs = llm.generate(prompts, sampling_config, use_tqdm=False)

        logger.info(f"Successfully generated outputs on GPU {gpu_id}.")
        return outputs

    except Exception as e:
        logger.error(f"Error during inference on GPU {gpu_id}: {str(e)}")
        raise


@validate_and_convert_configs(LLMParams, SamplingParams)
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
        sampling_config (SamplingParams): Parameters for sampling during inference.
        num_gpus (int): Number of GPUs to use for inference (default: 8).

    Returns:
        List[str]: A list of generated responses corresponding to the input prompts.
    """
    logger.info(f"Starting inference with {num_gpus} GPUs.")

    # Split the prompts equally among the available GPUs
    split_prompts = split_into_chunks(prompts, num_gpus)

    all_outputs = []

    # Run inference sequentially on each GPU
    for gpu_id, gpu_prompts in enumerate(split_prompts):
        logger.info(
            f"Running inference on GPU {gpu_id} with {len(gpu_prompts)} prompts."
        )

        try:
            # Run inference on the current GPU and collect the results
            output = generate_on_single_gpu(
                gpu_id, gpu_prompts, llm_config, sampling_config
            )
            all_outputs.extend(output)
            logger.info(f"Successfully generated outputs on GPU {gpu_id}.")

        except Exception as e:
            logger.error(f"Error on GPU {gpu_id}: {str(e)}")

    logger.info("Inference completed successfully across all GPUs.")

    # Extract and return the text from model outputs
    responses = [output.outputs[0].text for output in all_outputs]
    return responses
