for i in range(70+1):
    cmd1 = "cat " + "19-2-23-embedding-two-mimic-one-hot-mlp-" + str(i) + ".log? | grep final"
    cmd2 = "cat " + "19-2-23-embedding-two-mimic-one-hot-mlp-" + str(i) + ".log | grep final"
    echo_str = 'echo ' + '\"' + cmd1 + '\"'
    print (echo_str)
    print (cmd1)
    print (cmd2)
