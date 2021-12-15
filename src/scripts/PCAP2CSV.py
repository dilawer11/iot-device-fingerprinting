# Converts the PCAP Files to CSV Files for next steps

import sys
import os
import argparse
import csv
sys.path.append(os.getcwd())

class PCAP2CSVConverter:
    pcapDir = None
    csvDir = None
    CMD = '''tshark -r {} -T fields -e frame.number -e frame.time -e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e frame.len -e tcp.flags.ack -e tcp.flags.syn -e tcp.flags.fin -e tcp.flags.reset -e tcp.flags.push -e tcp.flags.urg -e _ws.col.Protocol -e eth.src -e eth.dst -E separator=,  -E quote=d > {}.csv'''
    def convertToCSV(self, pcapWithPath):
        print(pcapWithPath)
        filename, ext = os.path.splitext(pcapWithPath)
        filename = filename.replace(self.pcapDir, '')
        directory, _ = os.path.split(filename)
        if (directory[0]=='/' or directory[0] == '\\'):
            directory = directory[1:]
        complete_dir = os.path.join(self.csvDir, directory)
        if not os.path.isdir(os.path.join(self.csvDir, directory)):
            os.makedirs(os.path.join(self.csvDir, directory))
        if filename[0] == '/' or filename[0] == '\\':
            filename = filename[1:]
        destPath = os.path.join(self.csvDir, filename)
        cmd = self.CMD.format(pcapWithPath, destPath)
        os.system(cmd)
    def __init__(self, pcap_dir, csv_dir):
        self.pcapDir = pcap_dir
        self.csvDir = csv_dir
        if not os.path.isdir(self.csvDir):
            os.makedirs(self.csvDir)
    def start(self):
        fileList = []
        for root, dirs, files in os.walk(self.pcapDir):
            for name in files:
                if os.path.splitext(name)[1] == '.pcap':
                    fileList.append(os.path.join(root, name))
        count = 0
        print('Total Files:', len(fileList))
        for pcap in fileList:
            self.convertToCSV(pcap)
            count += 1
            print('.', end='')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_dir", help="Input Directory or the PCAP Directory")
    parser.add_argument("-o", dest="output_dir", help="Output Directory or the CSV Directory")
    args = parser.parse_args()
    if args.input_dir is None or args.output_dir is None:
        parser.print_help()
        return
    if not os.path.isdir(args.input_dir):
        print(f"No such directory: {args.input_dir}", file=sys.stderr)
        return
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    p2c = PCAP2CSVConverter(args.input_dir, args.output_dir)
    p2c.start()

if __name__ == "__main__":
    main()