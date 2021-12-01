

def inputfile2dict(inputfile, recognized_strings):
    """
    """
    out = {}
    
    with open(inputfile, 'r') as fh:
        for line in fh: 
            tmp = line.strip().split()
            print(tmp)
            if tmp == []:
                continue
            elif tmp[0] == '':
                continue
            elif tmp[0] in recognized_strings:
                identifier = tmp[0]
            else:
                if identifier not in out.keys():
                    out[identifier] = []
                print(len(tmp))
                if len(tmp) == 1:
                    out[identifier].append(tmp[0])
                else:
                    out[identifier].append(tmp)
    print(out)
    return out






