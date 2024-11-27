import multiprocessing
import os
from typing import Any, Dict, List, Optional

import numpy as np
from vllm import LLM

from .models import LLMParams, SamplingParams


def mgpu_inference(
    prompts: List[str],
    num_gpus: int = 8,
    llm_params: Optional[Dict] = None,
    sampling_params: Optional[Dict] = None,
) -> List[str]:
    """
    Interface for running multi-GPU inference with customizable parameters.

    Args:
        prompts (List[str]): A list of input prompts to generate responses for.
        num_gpus (int): Number of GPUs to use for inference (default: 8).
        llm_params (Optional[Dict]): Parameters for configuring the LLM model.
        sampling_params (Optional[Dict]): Parameters for sampling during inference.

    Returns:
        List[str]: A list of generated responses corresponding to the input prompts.
    """

    def generate_on_single_gpu(
        gpu_id: int,
        prompts: List[str],
        llm_config: LLMParams,
        sampling_config: SamplingParams,
    ) -> Any:
        """
        Internal function to run inference on a single GPU.

        Args:
            gpu_id (int): The GPU ID to use for inference.
            prompts (List[str]): A list of input prompts for the model.
            llm_config (LLMParams): Configuration for the LLM model.
            sampling_config (SamplingParams): Parameters for sampling during inference.

        Returns:
            Any: The generated outputs from the model.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        prompts_array = np.array(prompts)
        llm = LLM(
            model=llm_config.model,
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            trust_remote_code=llm_config.trust_remote_code,
        )
        return llm.generate(
            prompts_array, sampling_config.model_dump(), use_tqdm=False
        )

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
            data[
                i * len(data) // num_splits : (i + 1) * len(data) // num_splits
            ]
            for i in range(num_splits)
        ]

    # Validate and initialize LLM and Sampling parameters
    llm_config = LLMParams(**(llm_params or {}))
    sampling_config = SamplingParams(**(sampling_params or {}))

    # Split prompts among GPUs
    split_prompts = split_list(prompts, num_gpus)

    # Prepare inputs for each GPU process
    gpu_inputs = [
        (gpu_id, gpu_prompts, llm_config, sampling_config)
        for gpu_id, gpu_prompts in enumerate(split_prompts)
    ]

    # Run inference in parallel across GPUs
    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.starmap(generate_on_single_gpu, gpu_inputs)

    # Collect and combine results
    all_outputs = []
    for result in results:
        all_outputs.extend(result)

    # Extract the text from model outputs
    responses = [output.outputs[0].text for output in all_outputs]
    return responses


prompts = [
    "What is AI?",
    "Explain quantum computing.",
    "Define machine learning.",
]
llm_params = {
    "model": "gpt-4",
    "gpu_memory_utilization": 0.8,
    "trust_remote_code": True,
}
sampling_params = {"n": 2, "temperature": 0.7, "max_tokens": 256}

responses = mgpu_inference(
    prompts, num_gpus=4, llm_params=llm_params, sampling_params=sampling_params
)

for response in responses:
    print(response)
