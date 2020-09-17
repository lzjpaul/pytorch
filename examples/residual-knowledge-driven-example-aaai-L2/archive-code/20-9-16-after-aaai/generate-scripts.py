for i in range(184+1):
    cmd1 = "cat " + "8-28-fourth-run-average-" + str(i) + ".log? | grep final"
    cmd2 = "cat " + "8-28-fourth-run-average-" + str(i) + ".log | grep final"
    echo_str = 'echo ' + '\"' + cmd1 + '\"'
    print (echo_str)
    print (cmd1)
    print (cmd2)
