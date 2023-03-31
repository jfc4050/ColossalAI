from colossalai.logging import get_dist_logger
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = get_dist_logger()

class DummyPromptDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, size: int) -> None:
        super().__init__()
        logger.info("Loading (dummy) data...")
        self.size = size

        self.examples = []

        token = tokenizer(
            " ".join("blah" for _ in range(96)),
            return_tensors="pt",
            max_length=96,
            padding="max_length",
            truncation=True
        )
        for idx in token["input_ids"]:
            self.examples.append(idx.to(torch.cuda.current_device()))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Tensor:
        return self.examples[i % len(self.examples)]
