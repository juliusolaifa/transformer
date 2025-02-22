from tokenizers import Tokenizer
from typing import Optional, List, Union, Type, Iterable, Any
from pathlib import Path

def train_tokenizer(
        data: Union[list[str], Iterable[str]],
        model: Type,
        trainer: Type,
        pre_tokenizer: Type,
        special_tokens: List[str],
        unk_token: str,
        min_freq: int,
        normalizer: Optional[Any] = None,
        path: Optional[Union[str, Path]] = None
) -> Tokenizer:
    """Train a tokenizer with special configurations.

    Args:
        data: Training data (list or iterator os strings).
        model: Tokenizer model class (e.g., WordLevel, BPE).
        trainer: Trainer class (e.g., WordLevelTrainer, BpeTrainer).
        pre_tokenizer: Pre-tokenizer class (e.g., Whitespace).
        special_tokens: Additional special tokens (e.g., [CLS], [SEP]).
        unk_tokens: Unknown token (added as the first special token).
        min_freq: Minimum frequency for tokens to be included.
        normalizer: Optional normalizer (e.g., Lowercase).
        path: Optional path to save the trained tokenizer.

    Returns:
        Trained tokenizer instance
    """
    tokenizer = Tokenizer(model(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizer()
    trainer = trainer(special_tokens=special_tokens, min_freq=min_freq)
    if normalizer is not None:
        tokenizer.normalizer = normalizer
    tokenizer.train_from_iterator(data, trainer)
    if path is not None:
        tokenizer.save(path)
    return tokenizer

def get_tokenizer(path: Optional[Union[str, Path]]) -> Tokenizer:
    """Load a tokenizer from file

    Args:
        Path to load the tokenizer from

    Returns:
        A tokenizer instance
    """
    Tokenizer.from_file(str(path))
