import time
from crypto_utils import sha256_hash


class Block:

    def __init__(self,
                 index,
                 previous_hash,
                 parameter,
                 dataset_hash,
                 mae,
                 normalized_score,
                 merkle_root,
                 validator_signatures):

        self.index = index
        self.previous_hash = previous_hash
        self.parameter = parameter
        self.dataset_hash = dataset_hash
        self.mae = mae
        self.normalized_score = normalized_score
        self.merkle_root = merkle_root
        self.validator_signatures = validator_signatures
        self.timestamp = time.time()

    def compute_hash(self):

        block_data = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "parameter": self.parameter,
            "dataset_hash": self.dataset_hash,
            "mae": self.mae,
            "normalized_score": self.normalized_score,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp
        }

        return sha256_hash(block_data)