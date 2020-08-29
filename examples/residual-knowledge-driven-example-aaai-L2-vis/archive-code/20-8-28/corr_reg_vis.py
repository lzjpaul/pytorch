import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization for CORR Reg')
    parser.add_argument('-corr_file', type=str, help='correlation file')
    parser.add_argument('-weight_file', type=str, help='weight file')
    parser.add_argument('-output_dir', type=str, help='weight file')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir.rstrip()):
        os.mkdir(args.output_dir.rstrip())
        print("Directory " + args.output_dir.rstrip() +  " Created ")

    corr_abs = np.genfromtxt(args.corr_file, delimiter=',')
    w_array = np.genfromtxt(args.weight_file, delimiter=',')

    print ("corr_abs shape: ", corr_abs.shape)
    corr_abs_flat = corr_abs.reshape(-1)
    print ("corr_abs_flat shape: ", corr_abs_flat.shape)
    print ("w_array shape: ", w_array.shape)
    w_array_flat = w_array.reshape(-1)
    print ("w_array_flat shape: ", w_array_flat.shape)

    # corr_argsort_array = np.argsort(corr_abs_flat)  # ranking each element in terms of correlation values
    corr_argsort_array_index = np.argsort(np.argsort(corr_abs_flat))  # this one saves for each value of corr_abs_flat, what is the ranking?
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

    
    w_array_argsort_array_index = np.argsort(np.argsort(w_array_flat))  # this one saves for each value of corr_abs_flat, what is the ranking?
    w_array_argsort_array_index_normalize = w_array_argsort_array_index / float(num_elements)

    bottom_weight_values = w_array_flat[bottom_elements_index]
    top_weight_values = w_array_flat[top_elements_index]

    bottom_weight_ratios = w_array_argsort_array_index_normalize[bottom_elements_index]
    top_weight_ratios = w_array_argsort_array_index_normalize[top_elements_index]

    np.savetxt(args.output_dir.rstrip() + '/bottom_weight_values.csv', bottom_weight_values, fmt = '%6f', delimiter=",") #modify here
    np.savetxt(args.output_dir.rstrip() + '/top_weight_values.csv', top_weight_values, fmt = '%6f', delimiter=",") #modify here
    np.savetxt(args.output_dir.rstrip() + '/bottom_weight_ratios.csv', bottom_weight_ratios, fmt = '%6f', delimiter=",") #modify here
    np.savetxt(args.output_dir.rstrip() + '/top_weight_ratios.csv', top_weight_ratios, fmt = '%6f', delimiter=",") #modify here
