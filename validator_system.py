import random
from crypto_utils import generate_keypair


class Validator:

    def __init__(self, name, stake, reputation):

        self.name = name
        self.stake = stake
        self.reputation = reputation

        self.private_key, self.public_key = generate_keypair()

    def weight(self):

        return self.stake * self.reputation


def create_validators(num_nodes=5):

    validators = []

    for i in range(num_nodes):

        name = f"Node_{i+1}"

        stake = random.randint(50, 200)

        reputation = round(random.uniform(0.5, 1.5), 2)

        validators.append(Validator(name, stake, reputation))

    return validators


def select_validator(validators):

    weights = [v.weight() for v in validators]

    selected = random.choices(validators, weights=weights, k=1)[0]

    return selected