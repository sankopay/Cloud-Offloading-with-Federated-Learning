read_pickle(model.pickle)

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import multi_run_results_file_path, MAX_CASE

print(multi_run_results_file_path)
filename = multi_run_results_file_path


plt.figure(1)
for c in range(0, N_CASES):
    plt.semilogx(xaxis, [avg_list_loss[c][i] for i in fixed_local_it_indexes], label='Case' + str(c),
                 color=color_cases[c])
    plt.plot(tauAvg[c][tauAvgIndex], ([avg_list_loss[c][i] for i in adapt_local_it_indexes] * single_point)[0],
             marker='o', markersize=8, color=color_cases[c])

if loss_centralized is not None:
    plt.semilogx(xaxis, loss_centralized * single_point, '--', label='Decentralized case', color='black')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Value of \\tau')
plt.ylabel('Loss function Value (on Training Data)')

plt.figure(2)
for c in range(0, N_CASES):
    plt.semilogx(xaxis, [avg_list_acc[c][i] for i in fixed_local_it_indexes], label='Case' + str(c),
                 color=color_cases[c])
    plt.plot(tauAvg[c][tauAvgIndex], ([avg_list_acc[c][i] for i in adapt_local_it_indexes] * single_point)[0],
             marker='o', markersize=8, color=color_cases[c])

if accuracy_centralized is not None:
    plt.semilogx(xaxis, accuracy_centralized * single_point, '--', label='Decentralized case', color='black')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Value of \\tau')
plt.ylabel('Classification Accuracy (on Testing Data)')


plt.show()