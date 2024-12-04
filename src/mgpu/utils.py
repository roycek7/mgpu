from typing import Any, List


def split_into_chunks(data: List[Any], num_chunks: int) -> List[List[Any]]:
    """
    Split a list into approximately equal-sized chunks.

    Args:
        data (List[Any]): The list to split.
        num_chunks (int): The number of chunks.

    Returns:
        List[List[Any]]: A list of sublists, each containing a portion of the original data.
    """
    return [
        data[i * len(data) // num_chunks : (i + 1) * len(data) // num_chunks]
        for i in range(num_chunks)
    ]
