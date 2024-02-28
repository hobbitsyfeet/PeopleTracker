import datetime

##
# Simple print out that saves any messages into a file
def log(message, filename="Crashlog.txt"):
    '''
    Exports a message into Crashlog.txt
    Additionally prints message
    '''
    file = open(filename, 'a')
    print(message)
    file.write('['+str(datetime.datetime.now())+'] ' + str(message) + '\n')
    file.close()