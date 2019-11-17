import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: app_selector.py <mode>", file=sys.stderr)
        print(len(sys.argv))
        print(sys.argv[1] + '  ' + sys.argv[2])
        exit(-1)

    mode = sys.argv[1]

    if mode == 'antibreakin':
        exec(open('antibreakin/antibreakin.py').read())
    elif mode == 'userdetection':
        exec(open('userdetection/userdetection.py').read())
    elif mode == 'mooddetection':
        exec(open('mooddetection/src/mooddetection.py').read())
    else:
        print('Mode can be one of the following: antibreakin, userdetection, mooddetection')