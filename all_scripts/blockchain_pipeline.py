from crypto_utils import hash_file, sign_data
from merkle_tree import build_merkle_root
from validator_system import create_validators, select_validator
from blockchain_core import Blockchain
from block import Block
from network_simulator import simulate_network_delay

import json


def run_blockchain_pipeline(parameter, mae, score):

    dataset_file = f"{parameter}_dataset.json"

    prediction_file = f"{parameter}_prediction.json"

    dataset_hash = hash_file(dataset_file)

    prediction_hash = hash_file(prediction_file)

    merkle_root = build_merkle_root([prediction_hash])

    validators = create_validators(5)

    proposer = select_validator(validators)

    signatures = []

    block_data = parameter + dataset_hash + merkle_root

    for v in validators[:3]:

        simulate_network_delay()

        sig = sign_data(v.private_key, block_data)

        signatures.append({

            "validator": v.name,
            "public_key": v.public_key,
            "signature": sig
        })

    blockchain = Blockchain()

    previous_hash = blockchain.get_last_block()["hash"] if len(blockchain.chain) > 1 else "0"

    block = Block(
        index=len(blockchain.chain),
        previous_hash=previous_hash,
        parameter=parameter,
        dataset_hash=dataset_hash,
        mae=mae,
        normalized_score=score,
        merkle_root=merkle_root,
        validator_signatures=signatures
    )

    blockchain.add_block(block)

    print("Block added for", parameter)