import numpy as np
import matplotlib.pyplot as plt

# fake data
np.random.seed(19680801)
data = np.random.lognormal(size=(37, 10), mean=1.5, sigma=1.75)
labels = list('ABCDEFGHIJ')
fs = 10  # fontsize

print data.shape

fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True)
axes[0, 0].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[0, 0].set_title('Default', fontsize=fs)
axes[0, 0].set_ylabel('aaa')

axes[0, 1].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[0, 1].set_title('Default', fontsize=fs)

axes[0, 2].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[0, 2].set_title('Default', fontsize=fs)

axes[1, 0].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[1, 0].set_title('Default', fontsize=fs)

axes[1, 1].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[1, 1].set_title('Default', fontsize=fs)

axes[1, 2].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[1, 2].set_title('Default', fontsize=fs)

axes[2, 0].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[2, 0].set_title('Default', fontsize=fs)

axes[2, 1].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[2, 1].set_title('Default', fontsize=fs)

axes[2, 2].boxplot(data, labels=labels, showfliers=False, vert=False)
axes[2, 2].set_title('Default', fontsize=fs)

# axes[4, 1].boxplot(data, labels=labels, showfliers=False, vert=False)
# axes[4, 1].set_title('Default', fontsize=fs)

for ax in axes.flatten():
    ax.set_xscale('log')
    # ax.set_yticklabels([])

fig.subplots_adjust(hspace=0.4)
plt.show()
