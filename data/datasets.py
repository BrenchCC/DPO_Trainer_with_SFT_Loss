"""Dataset and collator implementations for SFT and DPO training."""

import json
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


def _load_json_data(data_path: str) -> List[Dict[str, Any]]:
    """Load a JSON array from disk.

    Args:
        data_path: Path to the JSON dataset.

    Returns:
        A list of dataset records.
    """
    with open(data_path, "r", encoding = "utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array: {data_path}")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"Every dataset record must be an object: {data_path}")

    return data


def _validate_model_max_length(model_max_length: int) -> None:
    """Validate the sequence length limit.

    Args:
        model_max_length: Maximum number of tokens in one sequence.
    """
    if model_max_length < 2:
        raise ValueError("model_max_length must be at least 2.")


def _normalize_messages(messages: Any, record_index: int) -> List[Dict[str, str]]:
    """Validate and normalize chat messages.

    Args:
        messages: Raw messages from one dataset record.
        record_index: Index used in validation errors.

    Returns:
        Normalized role-content message dictionaries.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {record_index}: messages must be a non-empty list.")

    normalized_messages = []
    valid_roles = {"system", "user", "assistant"}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Record {record_index}, message {message_index}: message must be an object."
            )

        role = message.get("role")
        content = message.get("content")
        if role not in valid_roles:
            raise ValueError(
                f"Record {record_index}, message {message_index}: invalid role {role!r}."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Record {record_index}, message {message_index}: content must be non-empty."
            )

        normalized_messages.append({"role": role, "content": content})

    return normalized_messages


def _normalize_history(history: Any, record_index: int) -> List[Dict[str, str]]:
    """Normalize DPO history into role-content messages.

    Args:
        history: A list of messages or user-assistant pairs.
        record_index: Index used in validation errors.

    Returns:
        Normalized chat messages.
    """
    if history in (None, []):
        return []
    if not isinstance(history, list):
        raise ValueError(f"Record {record_index}: history must be a list.")

    if all(isinstance(item, dict) for item in history):
        return _normalize_messages(history, record_index)

    messages = []
    for history_index, pair in enumerate(history):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"Record {record_index}, history {history_index}: expected a user-assistant pair."
            )

        user_content, assistant_content = pair
        if not isinstance(user_content, str) or not user_content.strip():
            raise ValueError(
                f"Record {record_index}, history {history_index}: user content must be non-empty."
            )
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            raise ValueError(
                f"Record {record_index}, history {history_index}: assistant content must be non-empty."
            )

        messages.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        )

    return messages


def _encode_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> List[int]:
    """Encode chat messages ending at the assistant generation marker.

    Args:
        tokenizer: Tokenizer providing apply_chat_template.
        messages: Prompt messages preceding the target response.

    Returns:
        Prompt token IDs.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("The tokenizer must provide apply_chat_template().")

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_dict = False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The tokenizer must define a usable chat template for SFT and DPO data."
        ) from error

    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("The tokenizer chat template produced no prompt tokens.")

    return prompt_ids


def _encode_response(tokenizer: Any, response: str) -> List[int]:
    """Encode one assistant response and append EOS.

    Args:
        tokenizer: Tokenizer providing encode and eos_token_id.
        response: Assistant response text.

    Returns:
        Response token IDs ending in EOS.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")

    response_ids = tokenizer.encode(response, add_special_tokens = False)
    if not isinstance(response_ids, list):
        response_ids = list(response_ids)
    if not response_ids or response_ids[-1] != eos_token_id:
        response_ids.append(eos_token_id)

    return response_ids


def _truncate_response(response_ids: List[int], max_response_length: int, eos_token_id: int) -> List[int]:
    """Right-truncate a response while retaining EOS.

    Args:
        response_ids: Response token IDs ending in EOS.
        max_response_length: Maximum retained response length.
        eos_token_id: Token ID used to terminate the response.

    Returns:
        Truncated response token IDs.
    """
    if len(response_ids) <= max_response_length:
        return response_ids
    if max_response_length == 1:
        return [eos_token_id]

    return response_ids[:max_response_length - 1] + [eos_token_id]


def _build_supervised_sample(
    tokenizer: Any,
    prompt_messages: List[Dict[str, str]],
    response: str,
    model_max_length: int
) -> Dict[str, List[int]]:
    """Build one prompt-response sample with masked prompt labels.

    Args:
        tokenizer: Tokenizer used for chat rendering and tokenization.
        prompt_messages: Messages preceding the assistant response.
        response: Assistant response supervised by the loss.
        model_max_length: Maximum sequence length.

    Returns:
        Input IDs, attention mask, and response-only labels.
    """
    prompt_ids = _encode_prompt(tokenizer, prompt_messages)
    response_ids = _encode_response(tokenizer, response)
    response_ids = _truncate_response(
        response_ids,
        model_max_length - 1,
        tokenizer.eos_token_id
    )

    prompt_budget = model_max_length - len(response_ids)
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids
    }


class SupervisedDataset(Dataset):
    """Dataset that expands every assistant turn into one SFT sample."""

    def __init__(self, data_path: str, tokenizer: Any, model_max_length: int):
        """Initialize an SFT dataset.

        Args:
            data_path: Path to a JSON array containing messages records.
            tokenizer: Tokenizer used to encode chat prompts and responses.
            model_max_length: Maximum sequence length.
        """
        _validate_model_max_length(model_max_length)
        self.samples = []

        for record_index, record in enumerate(_load_json_data(data_path)):
            messages = _normalize_messages(record.get("messages"), record_index)
            assistant_count = 0
            for message_index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                if message_index == 0:
                    raise ValueError(
                        f"Record {record_index}: an assistant message requires prior context."
                    )

                self.samples.append(
                    _build_supervised_sample(
                        tokenizer,
                        messages[:message_index],
                        message["content"],
                        model_max_length
                    )
                )
                assistant_count += 1

            if assistant_count == 0:
                raise ValueError(f"Record {record_index}: no assistant message found.")

    def __len__(self) -> int:
        """Return the number of expanded assistant samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """Return one encoded SFT sample.

        Args:
            index: Expanded sample index.

        Returns:
            Encoded model inputs and labels.
        """
        return self.samples[index]


class DPODataset(Dataset):
    """Dataset for paired chosen and rejected assistant responses."""

    def __init__(self, data_path: str, tokenizer: Any, model_max_length: int):
        """Initialize a DPO dataset.

        Args:
            data_path: Path to a JSON array containing preference records.
            tokenizer: Tokenizer used to encode prompts and responses.
            model_max_length: Maximum length of each paired sequence.
        """
        _validate_model_max_length(model_max_length)
        self.samples = []

        for record_index, record in enumerate(_load_json_data(data_path)):
            self.samples.append(
                self._encode_record(
                    record,
                    record_index,
                    tokenizer,
                    model_max_length
                )
            )

    @staticmethod
    def _encode_record(
        record: Dict[str, Any],
        record_index: int,
        tokenizer: Any,
        model_max_length: int
    ) -> Dict[str, Any]:
        """Encode one DPO preference record.

        Args:
            record: Raw preference record.
            record_index: Index used in validation errors.
            tokenizer: Tokenizer used for chat rendering and tokenization.
            model_max_length: Maximum length of each paired sequence.

        Returns:
            Encoded chosen and rejected sequences with response positions.
        """
        instruction = record.get("instruction")
        input_text = record.get("input", "")
        chosen = record.get("chosen")
        rejected = record.get("rejected")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Record {record_index}: instruction must be non-empty.")
        if not isinstance(input_text, str):
            raise ValueError(f"Record {record_index}: input must be a string.")
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError(f"Record {record_index}: chosen must be non-empty.")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"Record {record_index}: rejected must be non-empty.")

        user_content = instruction
        if input_text.strip():
            user_content = f"{instruction}\n{input_text}"

        prompt_messages = _normalize_history(record.get("history", []), record_index)
        prompt_messages.append({"role": "user", "content": user_content})
        prompt_ids = _encode_prompt(tokenizer, prompt_messages)

        chosen_ids = _encode_response(tokenizer, chosen)
        rejected_ids = _encode_response(tokenizer, rejected)
        chosen_ids = _truncate_response(
            chosen_ids,
            model_max_length - 1,
            tokenizer.eos_token_id
        )
        rejected_ids = _truncate_response(
            rejected_ids,
            model_max_length - 1,
            tokenizer.eos_token_id
        )

        prompt_budget = model_max_length - max(len(chosen_ids), len(rejected_ids))
        prompt_ids = prompt_ids[-prompt_budget:]
        response_start_position = len(prompt_ids) - 1
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_start_position": response_start_position,
            "rejected_input_ids": rejected_input_ids,
            "rejected_start_position": response_start_position,
            "labels": [-100] * len(prompt_ids) + chosen_ids
        }

    def __len__(self) -> int:
        """Return the number of preference records."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one encoded preference sample.

        Args:
            index: Preference sample index.

        Returns:
            Encoded chosen and rejected model inputs.
        """
        return self.samples[index]


class DPODataCollator:
    """Dynamically pad DPO pairs and chosen response labels."""

    def __init__(self, tokenizer: Any):
        """Initialize the DPO collator.

        Args:
            tokenizer: Tokenizer providing pad_token_id.
        """
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id.")

        self.pad_token_id = pad_token_id

    @staticmethod
    def _pad_sequences(sequences: Sequence[Sequence[int]], padding_value: int) -> torch.Tensor:
        """Right-pad integer sequences.

        Args:
            sequences: Variable-length integer sequences.
            padding_value: Value used for right padding.

        Returns:
            A padded long tensor.
        """
        max_length = max(len(sequence) for sequence in sequences)
        padded_sequences = [
            list(sequence) + [padding_value] * (max_length - len(sequence))
            for sequence in sequences
        ]

        return torch.tensor(padded_sequences, dtype = torch.long)

    @staticmethod
    def _build_attention_mask(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        """Build right-padded attention masks.

        Args:
            sequences: Variable-length integer sequences.

        Returns:
            A padded long attention mask.
        """
        max_length = max(len(sequence) for sequence in sequences)
        masks = [
            [1] * len(sequence) + [0] * (max_length - len(sequence))
            for sequence in sequences
        ]

        return torch.tensor(masks, dtype = torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate encoded preference records.

        Args:
            features: Encoded records from DPODataset.

        Returns:
            Padded chosen and rejected tensors, positions, and labels.
        """
        if not features:
            raise ValueError("DPODataCollator requires at least one feature.")

        chosen_sequences = [feature["chosen_input_ids"] for feature in features]
        rejected_sequences = [feature["rejected_input_ids"] for feature in features]
        label_sequences = [feature["labels"] for feature in features]

        return {
            "chosen_input_ids": self._pad_sequences(
                chosen_sequences,
                self.pad_token_id
            ),
            "chosen_attention_mask": self._build_attention_mask(chosen_sequences),
            "chosen_start_position": torch.tensor(
                [feature["chosen_start_position"] for feature in features],
                dtype = torch.long
            ),
            "rejected_input_ids": self._pad_sequences(
                rejected_sequences,
                self.pad_token_id
            ),
            "rejected_attention_mask": self._build_attention_mask(rejected_sequences),
            "rejected_start_position": torch.tensor(
                [feature["rejected_start_position"] for feature in features],
                dtype = torch.long
            ),
            "labels": self._pad_sequences(label_sequences, -100)
        }
