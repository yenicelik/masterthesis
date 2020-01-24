"""
    Utils for creating an experiments folder
"""
import os
import random
import string


def randomString(root_path="./", stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    rnd_str = ''.join(random.choice(letters) for i in range(stringLength))

    if not os.path.exists(rnd_str):
        os.makedirs(rnd_str)

    return rnd_str




