import time
from datetime import datetime
import sys


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


today = (str(datetime.today()))[11:19]
print("you excute this app at {}'O clock.".format(today))
h = int(today[0:2])
m = int(today[3:5])
s = int(today[6:8])
now_sec = h * 60 * 60 + m * 60 + s
put = input("Exit time: 1)5:30pm  2)8:30 : ")
if "1" == put:
    lim_sec = 62700 - now_sec
elif "2" == put:
    lim_sec = 73500 - now_sec

_ = 1
print("remaining time from {} is {} seconds.".format(today, lim_sec - _))
for i in range(0, lim_sec - _):
    time.sleep(1)
    printProgress(i, lim_sec, '{}seconds remain'.format(lim_sec - _), 'Complete', 1, 50)
    _ += 1
print("Congratulations!")
input()