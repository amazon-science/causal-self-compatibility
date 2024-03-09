# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    met = pd.read_csv('metrics.csv')
    for _ in range(2):
        plt.scatter(met['shd'], met['sid_lower'])
        plt.xlabel('SHD')
        plt.ylabel('SID Lower')
        plt.figure()
        plt.scatter(met['shd'], met['sid_upper'])
        plt.xlabel('SHD')
        plt.ylabel('SID Upper')
        plt.figure()
        met['sid_lower'] = np.random.permutation(met['sid_lower'])
        met['sid_upper'] = np.random.permutation(met['sid_upper'])

    plt.show()
