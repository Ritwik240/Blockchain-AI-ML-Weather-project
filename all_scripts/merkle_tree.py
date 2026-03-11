from crypto_utils import sha256_hash


def build_merkle_root(hash_list):

    if len(hash_list) == 1:
        return hash_list[0]

    new_level = []

    for i in range(0, len(hash_list), 2):

        left = hash_list[i]

        if i + 1 < len(hash_list):
            right = hash_list[i + 1]
        else:
            right = left

        combined = sha256_hash(left + right)

        new_level.append(combined)

    return build_merkle_root(new_level)