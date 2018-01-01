x=[1,2,0,-1]

def check(x,w=0,b=0):
    for each in x:
        y = w*each+b
        if y<0:
            return check(x,w+y*each,b+y)
    return w,b


print(check(x))