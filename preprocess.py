
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# train data
FTP72_data = pd.read_csv('drive_cycle_data/FTP72.csv')
FTP75_data = pd.read_csv('drive_cycle_data/FTP75.csv')
HUDDS_data = pd.read_csv('drive_cycle_data/HUDDS.csv')
CUEDC_data = pd.read_csv('drive_cycle_data/CUEDC-NCH.csv')

TRAIN_DATA = [
    FTP72_data,
    FTP75_data,
    HUDDS_data,
    CUEDC_data
]

x_train = []
y_train = []

for drive_cycle in TRAIN_DATA:
    for i, rows in drive_cycle.iterrows():
        x_row = [rows['Ref'], rows['Actual'], rows['Ref_Det']]
        y_row = [rows['Torque']]

        x_train.append(x_row)
        y_train.append(y_row)


# Normalization
# x_train = MinMaxScaler().fit_transform(x_train)
# y_train = MinMaxScaler().fit_transform(y_train)

# show data
plt.plot(x_train)
plt.show()
plt.plot(y_train)
plt.show()


with open('./drive_cycle_data/X_TRAIN.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ref', 'Actual', 'Ref_Det'])
    writer.writerows(x_train)

with open('./drive_cycle_data/Y_TRAIN.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Torque'])
    writer.writerows(y_train)


# test data
JC08_data = pd.read_csv('drive_cycle_data/JC08.csv')

x_test = []
y_test = []

for i, rows in JC08_data.iterrows():
    x_row = [rows['Ref'], rows['Actual'], rows['Ref_Det']]
    y_row = [rows['Torque']]

    x_test.append(x_row)
    y_test.append(y_row)

# Normalization
# x_train = MinMaxScaler().fit_transform(x_test)
# y_train = MinMaxScaler().fit_transform(y_test)

# show data
plt.plot(x_test)
plt.show()
plt.plot(y_test)
plt.show()

with open('./drive_cycle_data/X_TEST.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ref', 'Actual', 'Ref_Det'])
    writer.writerows(x_test)

with open('./drive_cycle_data/Y_TEST.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Torque'])
    writer.writerows(y_test)