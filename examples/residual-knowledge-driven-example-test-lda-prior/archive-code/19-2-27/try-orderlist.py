# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
from collections import OrderedDict

def add_module(name, module, modules):
    modules[name] = module
    print ('modules')
    print (modules)
    for key, value in modules.items() :
        print ('key,value')
        print (key, value)
    if name=='1':
        modules['1'][0]=modules['1'][0]+1
        print ('after add values modules')
        print (modules)
    

def sequential(*args):
    modules=OrderedDict()
    print ('args')
    print (args)
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        for key, module in args[0].items():
            add_module(key, module)
    else:
        for idx, module in enumerate(args):
            print ('idx')
            print (idx)
            add_module(str(idx), module, modules)

if __name__ == '__main__':
    a = [1,2,3,4]
    layers=[]
    layers.append(a)
    layers.append(a)
    sequential(*layers)
