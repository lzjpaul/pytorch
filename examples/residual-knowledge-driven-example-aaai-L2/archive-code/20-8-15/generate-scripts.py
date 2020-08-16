for i in range(125+1):
    cmd1 = "cat " + "20-8-5-first-run-solve-nan-" + str(i) + ".log? | grep final"
    cmd2 = "cat " + "20-8-5-first-run-solve-nan-" + str(i) + ".log | grep final"
    echo_str = 'echo ' + '\"' + cmd1 + '\"'
    print (echo_str)
    print (cmd1)
    print (cmd2)
