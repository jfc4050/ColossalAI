
from typing import Dict
from colossalai.logging import get_dist_logger
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .sft_dataset import preprocess

logger = get_dist_logger()

PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


class DummySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer, size: int):
        super().__init__()

        sources = [PROMPT_DICT["prompt_no_input"].format_map({"instruction": "blah"})]
        targets = [f"blah blah {tokenizer.eos_token}"]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_id, = data_dict["input_ids"]
        self.label, = data_dict["labels"]
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        return {"input_ids": self.input_id, "labels": self.label}
