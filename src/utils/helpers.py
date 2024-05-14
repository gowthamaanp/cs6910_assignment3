import time
import math

'''
Helper function to print time elapsed and estimated time remaining given the current time and progress %

Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))