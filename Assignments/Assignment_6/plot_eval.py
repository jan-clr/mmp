import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Read the tensorboard log from 'best_until_now.csv'
    df = pd.read_csv('best_until_now.csv')
    # Get the values from the dataframe
    steps = df['Step'].values
    values = df['Value'].values
    # Plot the values
    plt.plot(steps, values)
    plt.xlabel('Step')
    plt.ylabel('AP')
    plt.title('AP vs Step')
    plt.savefig('ap_vs_step.png')


if __name__ == '__main__':
    main()
