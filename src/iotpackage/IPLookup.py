import socket
import json
import os
import fcntl as F

# LOOKUPFILENAME = 'hostnameMap.json'
class IPLookup:
    __lookupMap = None
    sucessfull_getHostName = 0
    failed_calls_getHostName = 0
    lookup_file = None
    def __init__(self, plus=1, lookup_file=None):
        self.lookup_file = lookup_file
        if isinstance(self.lookup_file, str) and os.path.exists(self.lookup_file):
            with open(self.lookup_file, 'r') as f:
                F.flock(f, F.LOCK_SH)
                self.__lookupMap = json.load(f)
                F.flock(f, F.LOCK_UN)
        else:
            self.__lookupMap = {}
    def __del__(self):
        print('Updating Lookup File', flush=True)
        if os.path.exists(self.lookup_file):
            with open(self.lookup_file, 'r+') as f:
                F.flock(f, F.LOCK_EX)
                loadedMap = json.load(f)
                loadedMap.update(self.__lookupMap)
                f.seek(0)
                json.dump(loadedMap, f)
                F.flock(f, F.LOCK_UN)
        else:
            with open(self.lookup_file, 'w') as f:
                F.flock(f, F.LOCK_EX)
                json.dump(self.__lookupMap, f)
                F.flock(f, F.LOCK_UN)
        print('Done updating lookup file', flush=True)
        
    def __getIP(self, name):
        try:
            return socket.gethostbyname(name)
        except Exception as e:
            print('Exception occured at __getIP')
            return
    def __getHostName(self, ip):
        try:
            hostname = socket.getnameinfo((ip,0),0)[0]
            self.sucessfull_getHostName += 1
            return socket.getnameinfo((ip,0),0)[0]
        except Exception as e:
            self.failed_calls_getHostName += 1
            return ip
    def __getTLDplusByIP(self, ip):
        hostname = self.__getHostName(ip)
        if hostname == ip:
            return 'unknown'
        domains = hostname.split('.')
        if len(domains) > 3:
            plus = 3
        elif len(domains) >= 2:
            plus = 2
        else:
            return hostname
        joinOperator = '.'
        tldPlus = joinOperator.join(domains[-1 * plus:])
        return tldPlus
    def lookupHostName(self, ip):
        try:
            return self.__lookupMap[ip]
        except KeyError as e:
            tldPlus = self.__getTLDplusByIP(ip)
            self.__lookupMap[ip] = tldPlus
            return tldPlus
    def lookupIP(self, ip):
        ipArray = ip.split('.')
        ip3octets = '.'.join(ipArray[:3])
        return ip3octets

