import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # get all run cvs
    csvs = sorted(os.listdir('./run_csv'))

    plt.xlabel('Step')
    plt.ylabel('AP')

    for csv in csvs:
        # Read the tensorboard log from 'best_until_now.csv'
        df = pd.read_csv(f'./run_csv/{csv}')
        # Get the values from the dataframe
        steps = df['Step'].values
        values = df['Value'].values
        # Plot the values
        plt.plot(steps, values)

    plt.legend([csv.split('.')[0] for csv in csvs])
    plt.title('AP vs Step')
    plt.savefig('ap_vs_step.png', dpi=300)


if __name__ == '__main__':
    main()
