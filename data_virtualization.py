import matplotlib.pyplot as plt
import pandas as pd


dataset_path = 'dataset_fog_release'
fog_subject = 'S01R01'
control_subject = 'S04R01'

size = 65

fog_df = pd.read_csv(dataset_path + '/csv_files/' + fog_subject + '.csv')
fog_df = fog_df[fog_df.annontations == 2].head(size)
time_value = fog_df['Time'].to_list()[0]
fog_df = pd.read_csv(dataset_path + '/csv_files/' + fog_subject + '.csv')
fog_df = fog_df[fog_df.Time > time_value].head(size)

control_df = pd.read_csv(dataset_path + '/csv_files/' + control_subject + '.csv')
control_df = control_df[control_df.Time > time_value].head(size)

print(fog_df.shape)
print(control_df.shape)

nrows = 3
ncols = 3

fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)

for i, fname in enumerate(features):
  if fname != 'Time' and fname != class_var:
    plt.subplot(nrows, ncols, i)
    plt.tight_layout()
    plt.title("Time vs " + fname)
    plt.xlabel('Time (in ms)')
    plt.ylabel(fname)
    plt.plot(fog_df.Time, fog_df[fname])
    plt.plot(control_df.Time, control_df[fname])

plt.legend(["FOG", "Control",])
plt.show()