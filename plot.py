import matplotlib.pyplot as plt

x = range(8)

ISOLET_CSPA_NMI = [0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221]
ISOLET_CSPA_Accuracy = [0.5102, 0.5102, 0.5102, 0.5102, 0.5102, 0.5102, 0.5102, 0.5102]
ISOLET_KM_NMI = [0.695, 0.695, 0.695, 0.695, 0.695, 0.695, 0.695, 0.695]
ISOLET_KM_Accuracy = [0.478, 0.478, 0.478, 0.478, 0.478, 0.478, 0.478, 0.478]
ISOLET_SP_NMI = [0.7226, 0.7226, 0.7226, 0.7226, 0.7226, 0.7226, 0.7226, 0.7226]
ISOLET_SP_Accuracy = [0.498, 0.498, 0.498, 0.498, 0.498, 0.498, 0.498, 0.498]
ISOLET_SCSPA_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
ISOLET_SCSPA_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
ISOLET_SSP_NMI = [0.6753, 0.6910, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
ISOLET_SSP_Accuracy = [0.4227, 0.4531, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]

"""
MNIST-4000 Data
"""
MNIST_CSPA_NMI = [0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221]
MNIST_CSPA_Accuracy = [0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048]
MNIST_KM_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_KM_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SP_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SP_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SCSPA_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SCSPA_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SSP_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
MNIST_SSP_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]

"""
USPS Data
"""
USPS_CSPA_NMI = [0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221, 0.7221]
USPS_CSPA_Accuracy = [0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048, 0.6048]
USPS_KM_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_KM_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SP_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SP_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SCSPA_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SCSPA_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SSP_NMI = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
USPS_SSP_Accuracy = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]

y = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
yy = [0.65, 0.69, 0.70, 0.72, 0.73, 0.73, 0.72, 0.73]
dataset_name = ['ISOLET-5', 'MNIST4000', 'USPS']
comparision = ['KM', 'CSPA', 'SP', 'SCSPA', 'SSP']

data = [
        # ISOLET-5 NMI
        {'KM': [x, ISOLET_KM_NMI],
         'CSPA': [x, ISOLET_CSPA_NMI],
         'SP': [x, ISOLET_SP_NMI],
         'SCSPA': [x, ISOLET_SCSPA_NMI],
         'SSP': [x, ISOLET_SSP_NMI]},
        # ISOLET-5 Accuracy
        {'KM': [x, ISOLET_KM_Accuracy],
         'CSPA': [x, ISOLET_CSPA_Accuracy],
         'SP': [x, ISOLET_SP_Accuracy],
         'SCSPA': [x, ISOLET_SCSPA_Accuracy],
         'SSP': [x, ISOLET_SSP_Accuracy]},
        # NMIST4000 NMI
        {'KM': [x, MNIST_KM_NMI],
         'CSPA': [x, MNIST_CSPA_NMI],
         'SP': [x, MNIST_SP_NMI],
         'SCSPA': [x, MNIST_SCSPA_NMI],
         'SSP': [x, MNIST_SSP_NMI]},
        # MNIST4000 Accuracy
        {'KM': [x, MNIST_KM_Accuracy],
         'CSPA': [x, MNIST_CSPA_Accuracy],
         'SP': [x, MNIST_SP_Accuracy],
         'SCSPA': [x, MNIST_SCSPA_Accuracy],
         'SSP': [x, MNIST_SSP_Accuracy]},
        # USPS NMI
        {'KM': [x, USPS_KM_NMI],
         'CSPA': [x, USPS_CSPA_NMI],
         'SP': [x, USPS_SP_NMI],
         'SCSPA': [x, USPS_SCSPA_NMI],
         'SSP': [x, USPS_SSP_NMI]},
        # USPS Accuracy
        {'KM': [x, USPS_KM_Accuracy],
         'CSPA': [x, USPS_CSPA_Accuracy],
         'SP': [x, USPS_SP_Accuracy],
         'SCSPA': [x, USPS_SCSPA_Accuracy],
         'SSP': [x, USPS_SSP_Accuracy]}
        ]

ticklabels = [
              # ISOLET-5
              ('1', '2', '3', '44', '55', '666', '777', '888', '999', '000'),
              # MNIST-4000
              ('1', '2', '3', '44', '55', '666', '777', '888', '999', '000'),
              # USPS
              ('1', '2', '3', '44', '55', '666', '777', '888', '999', '000')
             ]

marker = {'KM': '*',
          'CSPA': 'v',
          'SP': 'o',
          'SCSPA': 'D',
          'SSP': 's'}

colors = {'KM': 'red',
          'CSPA': 'magenta',
          'SP': 'blue',
          'SCSPA': 'green',
          'SSP': 'orange'}

fig, ax = plt.subplots(nrows=3, ncols=2)
count = 0
for row in ax:
    print type(row)
    for col in row:
        count += 1
        for cmp in comparision:
            col.plot(data[count-1][cmp][0], data[count-1][cmp][1], label=cmp, marker=marker[cmp], color=colors[cmp])
        col.set_title('$'+dataset_name[(count-1)/2]+'$')
        if count >=5:
            col.set_xlabel('#representative points')
        col.set_ylabel('NMI' if count % 2 != 0 else 'Accuracy')
        col.set_xticklabels(ticklabels[(count-1)/2], fontsize=10)
        col.legend(loc='lower right', fontsize=9)

fig.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.32)
for row in ax:
    for col in row:
        col.set_yticklabels(col.get_yticklabels(), fontsize=10)
plt.show()

