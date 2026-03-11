import json
import os
from block import Block


CHAIN_FILE = "weather_chain_pos.json"


class Blockchain:

    def __init__(self):

        if not os.path.exists(CHAIN_FILE):

            self.chain = []

            self.create_genesis_block()

        else:

            with open(CHAIN_FILE) as f:
                self.chain = json.load(f)

    def create_genesis_block(self):

        genesis = {

            "index": 0,
            "previous_hash": "0",
            "parameter": "GENESIS",
            "dataset_hash": "0",
            "mae": 0,
            "normalized_score": 0,
            "merkle_root": "0",
            "validator_signatures": [],
            "timestamp": 0
        }

        self.chain.append(genesis)

        self.save_chain()

    def get_last_block(self):

        return self.chain[-1]

    def add_block(self, block):

        block_hash = block.compute_hash()

        block_dict = block.__dict__

        block_dict["hash"] = block_hash

        self.chain.append(block_dict)

        self.save_chain()

    def save_chain(self):

        with open(CHAIN_FILE, "w") as f:

            json.dump(self.chain, f, indent=4)