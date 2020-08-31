import datetime

def log(message, filename="Crashlog.txt"):
    file = open(filename, 'a')
    print(message)
    file.write('['+str(datetime.datetime.now())+'] ' + str(message) + '\n')
    file.close()