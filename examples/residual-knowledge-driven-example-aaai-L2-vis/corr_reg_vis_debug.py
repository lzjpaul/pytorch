import numpy as np
import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Visualization for CORR Reg')
    # parser.add_argument('-corr_file', type=str, help='correlation file')
    # parser.add_argument('-weight_file', type=str, help='weight file')
    # args = parser.parse_args()

    # corr_abs = np.genfromtxt(args.corr_file, delimiter=',')
    # w_array = np.genfromtxt(args.weight_file, delimiter=',')
    corr_abs = np.array([[5,3,2],[1,7,9],[4,6,8]])
    w_array = np.array([[0.5,0.3,0.2],[0.1,0.7,0.9],[0.4,0.6,0.8]])

    print ("corr_abs shape: ", corr_abs.shape)
    corr_abs_flat = corr_abs.reshape(-1)
    print ("corr_abs_flat shape: ", corr_abs_flat.shape)
    print ("corr_abs_flat: ", corr_abs_flat)
    print ("w_array shape: ", w_array.shape)
    w_array_flat = w_array.reshape(-1)
    print ("w_array_flat shape: ", w_array_flat.shape)
    print ("w_array_flat: ", w_array_flat)

    # corr_argsort_array = np.argsort(corr_abs_flat)  # ranking each element in terms of correlation values
    corr_argsort_array_index = np.argsort(np.argsort(corr_abs_flat))  # this one saves for each value of corr_abs_flat, what is the ranking?
    print ("corr_argsort_array_index: ", corr_argsort_array_index)
    num_elements = corr_abs_flat.shape[0]
    print ("num_elements: ", num_elements)
    num_selected_elements = int(num_elements * 0.25)
    print ("num_selected_elements: ", num_selected_elements)
    """
    corr_argsort_array_index = np.zeros(num_elements)  # this one saves for each value of corr_abs_flat, what is the ranking?
    for i in range(num_elements):
        corr_argsort_array_index[i] = np.where(corr_argsort_array==i)[0][0]
    """
    bottom_elements_index = np.argwhere(corr_argsort_array_index < num_selected_elements)
    top_elements_index = np.argwhere(corr_argsort_array_index >= (num_elements - num_selected_elements))
    print ("bottom_elements_index: ", bottom_elements_index)
    print ("top_elements_index: ", top_elements_index)

    
    w_array_argsort_array_index = np.argsort(np.argsort(w_array_flat))  # this one saves for each value of corr_abs_flat, what is the ranking?
    print ("w_array_argsort_array_index: ", w_array_argsort_array_index)
    w_array_argsort_array_index_normalize = w_array_argsort_array_index / float(num_elements)
    print ("w_array_argsort_array_index_normalize: ", w_array_argsort_array_index_normalize)

    bottom_weight_values = w_array_flat[bottom_elements_index]
    top_weight_values = w_array_flat[top_elements_index]
    print ("bottom_weight_values: ", bottom_weight_values)
    print ("top_weight_values: ", top_weight_values)

    bottom_weight_ratios = w_array_argsort_array_index_normalize[bottom_elements_index]
    top_weight_ratios = w_array_argsort_array_index_normalize[top_elements_index]
    print ("bottom_weight_ratios: ", bottom_weight_ratios)
    print ("top_weight_ratios: ", top_weight_ratios)

    np.savetxt('bottom_weight_values.csv', bottom_weight_values, fmt = '%6f', delimiter=",") #modify here
    np.savetxt('top_weight_values.csv', top_weight_values, fmt = '%6f', delimiter=",") #modify here
    np.savetxt('bottom_weight_ratios.csv', bottom_weight_ratios, fmt = '%6f', delimiter=",") #modify here
    np.savetxt('top_weight_ratios.csv', top_weight_ratios, fmt = '%6f', delimiter=",") #modify here
