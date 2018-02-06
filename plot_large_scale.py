import matplotlib.pyplot as plt

x = range(10)
y = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
yy = [0.65, 0.69, 0.70, 0.72, 0.73, 0.73, 0.72, 0.73, 0.72, 0.715]
dataset_name = ['MNIST', 'SKIN', 'Covtype']
comparision = ['KM', 'SCSPA', 'SSP']
data = {'KM': [x, y],
        'SCSPA': [x, y],
        'SSP': [x, y]}
marker = {'KM': '*',
          'SCSPA': 'x',
          'SSP': 'D'}
fig, ax = plt.subplots(nrows=3, ncols=2)
count = 0
for row in ax:
    print type(row)
    for col in row:
        count += 1
        for cmp in comparision:
            if cmp == 'SSP':
                col.plot(data[cmp][0], yy, label=cmp, marker=marker[cmp])
            else:
                col.plot(data[cmp][0], data[cmp][1], label=cmp, marker=marker[cmp])
        col.set_title('$'+dataset_name[(count-1)/2]+'$')
        if count >=5:
            col.set_xlabel('$number\ of\ representatives$')
        col.set_ylabel('NMI' if count % 2 != 0 else 'Accuracy')
        col.set_xticklabels(('1', '2', '3', '44', '55', '666', '777', '888', '999', '000'), fontsize=10)
        # col.legend(loc='lower right', fontsize=10)

fig.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.32)
for row in ax:
    for col in row:
        col.set_yticklabels(col.get_yticklabels(), fontsize=10)
plt.legend()
plt.show()

