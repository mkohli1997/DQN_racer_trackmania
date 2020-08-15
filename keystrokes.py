import win32api as wapi
import time

keyList = []
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys