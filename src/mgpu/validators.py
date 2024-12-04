from functools import wraps
from typing import Any, Callable, List, Type


def validate_and_convert_configs(llm_model: Type, sampling_model: Type) -> Callable:
    """
    Decorator to validate and convert dictionary parameters into Pydantic models.

    Args:
        llm_model (Type): The Pydantic model class for LLM parameters.
        sampling_model (Type): The Pydantic model class for sampling parameters.

    Returns:
        Callable: A decorator for functions that validates and converts configs.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            prompts: List[str],
            llm_config: Any,
            sampling_config: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Validate and convert llm_config
            if isinstance(llm_config, dict):
                llm_config = llm_model(**llm_config)
            elif not isinstance(llm_config, llm_model):
                raise TypeError(
                    f"Expected llm_config to be of type {llm_model.__name__} or dict, "
                    f"but got {type(llm_config).__name__}."
                )

            # Validate and convert sampling_config
            if isinstance(sampling_config, dict):
                sampling_config = sampling_model(**sampling_config)
            elif not isinstance(sampling_config, sampling_model):
                raise TypeError(
                    f"Expected sampling_config to be of type {sampling_model.__name__} or dict, "
                    f"but got {type(sampling_config).__name__}."
                )

            # Call the original function with the validated models
            return func(prompts, llm_config, sampling_config, *args, **kwargs)

        return wrapper

    return decorator
