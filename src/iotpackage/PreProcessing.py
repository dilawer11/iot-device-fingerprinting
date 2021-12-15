import numpy as np
import pandas as pd
import os

class PreProcessor:
    printDetails = False
    srcName = None
    dstName = None
    colNameMappings = None
    deviceMappings = None
    winSize = None
    colDeviceLabel = None
    non_IoT = []
    packet_padding_counter_measure = False
    def __init__(self, src_name='SrcIP', dst_name='DstIP', col_name_mappings='IP', device_mappings=None, win_size='300s', col_device_label='Device', print_details=False, packet_padding=False):
        self.printDetails = print_details
        self.srcName = src_name
        self.dstName = dst_name
        self.colNameMappings = col_name_mappings
        self.winSize = win_size
        self.colDeviceLabel = col_device_label
        self.packet_padding_counter_measure = packet_padding
        if not isinstance(device_mappings, pd.DataFrame): raise ValueError(f'device_mappings: Expected a pandas.DataFrame instead of:{type(device_mappings)}')
        self.deviceMappings = device_mappings
    def dropEmptyIP(self, packets):
        nonIPPackets = (packets['SrcIP'].isna()) | (packets['DstIP'].isna())
        IPPackets = ~nonIPPackets
        return packets[IPPackets]
    def checkDouble(self, packet):
        try:
            if len(packet[self.srcName].split(',')) == 2:
                return True
            else:
                return False
        except:
            return False
    def __checkIP(self, ip):
        try:
            splited = ip.split(',')
            if (len(splited) == 2):
                return splited[0]
            else:
                return ip
        except AttributeError:
            return ip
    def __cleanICMP(self, packets):
        ICMPPackets = packets['Protocol'] == 'ICMP'
        if self.printDetails:
            temp = packets.apply(self.checkDouble, axis=1)
            print('Double IP Packets:', packets.shape[0])
        packets.loc[ICMPPackets, self.srcName] = packets.loc[ICMPPackets, self.srcName].apply(self.__checkIP)
        packets.loc[ICMPPackets, self.dstName] = packets.loc[ICMPPackets, self.dstName].apply(self.__checkIP)
        if self.printDetails:
            print('Removed Double IP Packets')
            temp = packets.apply(self.checkDouble, axis=1)
            print('Double IP Packets:', packets.shape[0])
        return packets
    def __dropExtraCols(self, packets):
        oldShape = packets.shape
        if self.colNameMappings != 'IP':
            packets = packets.drop(columns=['tcpSrcPort','udpSrcPort','tcpDstPort','udpDstPort','Proto', 'SrcIP', 'DstIP', self.srcName, self.dstName])
        else:
            packets = packets.drop(columns=['tcpSrcPort','udpSrcPort','tcpDstPort','udpDstPort','Proto', self.srcName, self.dstName])
        if self.printDetails: print('Dropped Extra Columns, Old Shape: {}, New Shape, {}'.format(oldShape[1], packets.shape[1]))
        return packets
    def __combinePorts(self, packets):
        packets.loc[:,'SrcPort'] = packets['tcpSrcPort'].combine_first(packets['udpSrcPort'])
        packets.loc[:,'DstPort'] = packets['tcpDstPort'].combine_first(packets['udpDstPort'])
        return packets
    def __assignPacketTypeandIP(self, packets):
        if self.printDetails: print('Initial Packets {}'.format(packets.shape))
        srcPackets = packets[self.srcName].isin(self.deviceMappings[self.colNameMappings])
        dstPackets = packets[self.dstName].isin(self.deviceMappings[self.colNameMappings])
        
        try:
            packets.loc[srcPackets, 'PacketType'] = 'Outgoing'
        except ValueError as e:
            print('ValueError @ __assignPacketTypeandIP', 'srcPacketsShape:',packets[srcPackets].shape)
        try:
            packets.loc[dstPackets, 'PacketType'] = 'Incoming'
        except ValueError as e:
            print('ValueError @ __assignPacketTypeandIP', 'dstPacketsShape:',packets[dstPackets].shape)
        packets.loc[srcPackets, 'IP'] = packets.loc[srcPackets, 'DstIP']
        packets.loc[dstPackets, 'IP'] = packets.loc[dstPackets, 'SrcIP']
        
        localTraffic = srcPackets & dstPackets
        noiseTraffic = (~srcPackets) & (~dstPackets)
        unwantedTraffic = localTraffic | noiseTraffic
        packets = packets[~unwantedTraffic]
        
        if self.printDetails: print('Filtered Packets {}'.format(packets.shape))
        return packets
    def __assignDeviceLabelPerDevice(self, device, packets):
        deviceName = device[self.colDeviceLabel]
        deviceID = device[self.colNameMappings]
        if deviceName in self.non_IoT:
            deviceName = 'Non-IoT'
        outPackets = packets[self.srcName] == deviceID
        inPackets = packets[self.dstName] == deviceID
        idx_bool_series = outPackets | inPackets
        try:
            packets.loc[idx_bool_series, self.colDeviceLabel] = deviceName
        except ValueError:
            if packets[idx_bool_series].shape[0] != 0 or packets[idx_bool_series].shape[1] != 0:
                print('ValueError @ __assignDeviceLabelPerDevice', 'packetsShape:', packets[idx_bool_series].shape)
    def __assignDeviceLabels(self, packets):
        if self.printDetails: print('Assigning Device Labels')
        self.deviceMappings.apply(self.__assignDeviceLabelPerDevice, args=(packets,), axis=1)
        return packets
    def __convertToTime(self, packets):
        if self.printDetails: print('Converting Time') 
        try:
            packets['Time'] = pd.to_datetime(packets['Time'])
        except Exception as e:
            print('ERROR:', e)
            packets['Time'] = packets['Time'].astype('datetime64')
        packets['TimeStamp'] = packets['Time']
        return packets
    def __sortPackets(self, packets, by="Time"):
        if self.printDetails: print('Sorting Packets by', by)
        return packets.sort_values(by=by)

    def run(self, packets, convert_time=True):
        packets = self.__cleanICMP(packets)
        if self.colNameMappings != 'IP':
            packets = self.dropEmptyIP(packets)
        packets = self.__combinePorts(packets)
        packets = self.__assignPacketTypeandIP(packets)
        packets = self.__assignDeviceLabels(packets)
        packets = self.__dropExtraCols(packets)
        if convert_time:
            packets = self.__convertToTime(packets)
            packets = self.__sortPackets(packets)
        else:
            packets = self.__sortPackets(packets, by="Frame")
        return packets
