import matplotlib.pyplot as plt


def _build_dict(data, cmps):
    if data.shape[0] != len(cmps):
        raise ValueError('[PLOT] Data Provided are not matched with the number of comparisons.')
    set_dict = {}
    for cmp, data_line in zip(cmps, data):
        set_dict[cmp] = data_line
    return set_dict


def plot(comparisons, raw_data, colors, markers, linestyles, titles, xlabel, ylabel, xticks, ncol, nrow, same_tick=True,
         markersize=6, markerwidth=1, linewidth=1, tfontsize=10):
    data = []
    if len(raw_data) != ncol * nrow:
        raise ValueError('[PLOT] Data Provided are not matched with ncol and nrow.')
    for raw_data_per_set in raw_data:
        set_dict = _build_dict(raw_data_per_set, comparisons)
        data.append(set_dict)
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(7, 10))
    count = 0
    xticks_data = range(raw_data[0].shape[1])
    print xticks_data
    for row in ax:
        for col in row:
            count += 1
            for cmp in comparisons:
                col.plot(xticks_data, data[count - 1][cmp], label=cmp, marker=markers[cmp],
                         color=colors[cmp], ms=markersize, mew=markerwidth, lw=linewidth, linestyle=linestyles[cmp])
            col.set_title(titles[count - 1], fontsize=10)
            if count == 1:
                col.set_ylim(0.4, 1.0)
            if (count - 1) % ncol == 0:
                col.set_ylabel(ylabel)
            if count > (nrow - 1) * ncol:
                col.set_xlabel(xlabel)
            if same_tick:
                col.set_xticks(xticks_data)
                col.set_xticklabels(xticks, fontsize=tfontsize)
            else:
                col.set_xticklabels(xticks[count - 1], fontsize=tfontsize)
            h, l = col.get_legend_handles_labels()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.32)
    for row in ax:
        for col in row:
            col.set_yticklabels(col.get_yticklabels(), fontsize=tfontsize)
    led = fig.legend(h, l, loc=(0.01, 0), mode='expand', ncol=len(comparisons)/2, fontsize=10)
    led.get_frame().set_edgecolor('white')
    plt.show()
    return


def plot_boxplot(raw_data, titles, ncol, nrow, xlabel, ylabel, titlesize=10):
    count = 0
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # labels = ['Weight(METIS)', 'Weight(Spec)', 'Prop(METIS)', 'Prop(SPEC)', 'E2CP', 'Cop-KMeans',
    #                'E2CP(METIS)', 'E2CP(Spec)', 'Cop-KMeans(METIS)', 'Cop-KMeans(Spec)']
    print labels
    labels = labels[0:raw_data[0].shape[0]]
    print labels
    fig, axes = plt.subplots(nrows=ncol, ncols=nrow, sharey=True, figsize=(6.5, 10))
    for i in range(0, nrow):
        for j in range(0, ncol):
            print raw_data[count].shape
            print len(labels)
            axes[i, j].boxplot(raw_data[count].transpose(), labels=labels, showfliers=False, vert=False)
            axes[i, j].set_title(titles[count], fontsize=titlesize)
            count += 1
            ticks = axes[i, j].get_xticks()
            axes[i, j].set_xticklabels(ticks, fontsize=9)
            if (count - 1) % ncol == 0:
                axes[i, j].set_ylabel(ylabel)
            if count > (nrow - 1) * ncol:
                axes[i, j].set_xlabel(xlabel)
            # if count == 1:
            #     axes[i, j].set_xlim(0.5, 1.0)
            #     axes[i, j].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            #     axes[i, j].set_xticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9)
            # if count == 8:
            #     axes[i, j].set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # plt.savefig()
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.95, wspace=0.16)
    plt.savefig('test.png', dpi=500)
    plt.show()
    return

if __name__ == '__main__':
    plt.plot([1, 2, 3], [1, 2, 3], 'x-', ms=5, mew=3, label='line 1', lw=1)
    plt.xticks([1, 2, 3])
    plt.show()
