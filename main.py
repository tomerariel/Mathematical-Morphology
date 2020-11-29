flag = True
while flag:
    ex = input('Which example would you like to view?\nEx.a: Edge detection 1\nEx.b: Edge detection 2\n\
Ex.c: Noise removal\nEx.d: Object decomposition 1\nEx.e: Object decomposition 2\nEx.f: Shape correction\n\
To exit, press 0.')

    if ex == 'a':
        scr = 'edge detection 1.py'
    elif ex == 'b':
        scr = 'edge detection 2.py'
    elif ex == 'c':
        scr = 'noise removal 1.py'
    elif ex == 'd':
        scr = 'object decomposition 1.py'
    elif ex == 'e':
        scr = 'object decomposition 2.py'
    elif ex == 'f':
        scr = 'shape correction 1.py'
    elif ex == '0':
        print('Thank you for using our morphology library.\nHave a lovely day :)')
        flag = False
        break
    else:
        print('Please choose one of the following examples or press 0 to exit')

    exec(open(scr).read())
