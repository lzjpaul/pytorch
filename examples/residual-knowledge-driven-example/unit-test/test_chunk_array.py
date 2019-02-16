import numpy as np

def chunk_array(arr, chunks, dim):
    if dim == 0:
        chunk_array_list = []
        base = int(arr.shape[0] / chunks)
        for i in range(chunks):
            chunk_array_list.append(arr[i * base: (i+1) * base])
    return chunk_array_list


if __name__ == '__main__':
    w_array = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    chunk_array = chunk_array(w_array,4,0)
    print ('chunk_array: \n', chunk_array)
    for i in range(4):
        print ('chunk_array[i] shape: \n', chunk_array[i].shape)
        print ('chunk_array[i]: \n', chunk_array[i])
