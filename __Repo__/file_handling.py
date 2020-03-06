import os
import numpy as np
import pandas as pd
from collections import defaultdict


"""
https://stackoverflow.com/questions/1724693/find-a-file-in-python
https://www.geeksforgeeks.org/os-walk-python/
"""

def findFile(fileName):
    for root, dirs, files in os.walk("../../", topdown=True):
        if fileName in files:
            return os.path.join(root, fileName)
    # If file is not found:
    raise FileNotFoundError("File  \'{}\'  Not Found".format(fileName))


def main():
    df = pd.read_csv(findFile("fileName"))
    df.head()


if __name__ == "__main__":
    main()
