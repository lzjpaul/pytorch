for i in range(49):
    cmd = "cat " + "19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-" + str(i) + ".log? | grep final"
    echo_str = 'echo ' + '\"' + cmd + '\"'
    print (echo_str)
    print (cmd)
