import numpy as np
from scipy.stats import pearsonr
import math

def pearson_correlation(numbers_x, numbers_y):
    mean_x = sum(numbers_x)/len(numbers_x)
    mean_y = sum(numbers_y)/len(numbers_y)

    subtracted_mean_x = [i - mean_x for i in numbers_x]
    subtracted_mean_y = [i - mean_y for i in numbers_y]

    x_times_y = [a * b for a, b in list(zip(subtracted_mean_x, subtracted_mean_y))]

    x_squared = [i * i for i in numbers_x]
    y_squared = [i * i for i in numbers_y]

    return sum(x_times_y) / math.sqrt(sum(x_squared) * sum(y_squared))

if __name__ == '__main__':
    # Contrary to what I said above about variables in general,
    # the names X and Y are ok here because they match the names in the formula.
    X = [12, 11, 13, 13, 9, 10, 10, 13, 5, 10, 10, 13, 10, 10, 5, 8, 9, 8, 8, 9, 9, 10, 11, 5, 12]
    Y = [11, 10, 10, 10, 9, 13, 10, 11, 6, 7, 13, 14, 14, 11, 11, 10, 10, 7, 8, 12, 11, 11, 8, 7, 13]

    print('manually: ', pearson_correlation(X, Y))
    r, p = pearsonr(np.array(X),np.array(Y))
    print ('scipy: ', r)
