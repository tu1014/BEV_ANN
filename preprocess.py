
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# train data
FTP72_data = pd.read_csv('drive_cycle_data/FTP72.csv')
FTP75_data = pd.read_csv('drive_cycle_data/FTP75.csv')
HUDDS_data = pd.read_csv('drive_cycle_data/HUDDS.csv')
JC08_data = pd.read_csv('drive_cycle_data/JC08.csv')

DRIVE_CYCLE_LIST = [
    FTP72_data,
    FTP75_data,
    HUDDS_data,
    JC08_data
]

x_train = []
y_train = []

x_test = []
y_test = []

for drive_cycle in DRIVE_CYCLE_LIST:

    length = len(drive_cycle)
    flag = length / 4 * 3

    for i, rows in drive_cycle.iterrows():
        x_row = [rows['Ref'], rows['Ref_d'], rows['Actual'], rows['Actual_d']]
        y_row = [rows['Torque']]

        if i < flag:
            x_train.append(x_row)
            y_train.append(y_row)
        else:
            x_test.append(x_row)
            y_test.append(y_row)


print('X_TRAIN SIZE : ', len(x_train))
print('Y_TRAIN SIZE : ', len(y_train))
print('X_TEST SIZE : ', len(x_test))
print('Y_TEST SIZE : ', len(y_test))

# Normalization
x_train = MinMaxScaler().fit_transform(x_train)
# y_train = MinMaxScaler().fit_transform(y_train)

x_test = MinMaxScaler().fit_transform(x_test)
# y_test = MinMaxScaler.fit_transform(y_test)

# show data
plt.subplot(4, 1, 1)
plt.plot(x_train)
plt.title('x_train')

plt.subplot(4, 1, 2)
plt.plot(y_train)
plt.title('y_train')

plt.subplot(4, 1, 3)
plt.plot(x_test)
plt.title('x_test')

plt.subplot(4, 1, 4)
plt.plot(y_test)
plt.title('y_test')

plt.show()


with open('./drive_cycle_data/X_TRAIN_NORMALIZATION.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ref', 'Ref_d', 'Actual', 'Actual_d'])
    writer.writerows(x_train)

with open('./drive_cycle_data/Y_TRAIN_NORMALIZATION.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Torque'])
    writer.writerows(y_train)

with open('./drive_cycle_data/X_TEST_NORMALIZATION.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ref', 'Ref_d', 'Actual', 'Actual_d'])
    writer.writerows(x_test)

with open('./drive_cycle_data/Y_TEST_NORMALIZATION.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Torque'])
    writer.writerows(y_test)

print('FINISH')
