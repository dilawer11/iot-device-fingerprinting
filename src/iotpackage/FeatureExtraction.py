import os
import numpy as np
import pandas as pd
import json
from iotpackage.Utils import getFeatureNames
from iotpackage.PreProcessing import PreProcessor
from iotpackage.IPLookup import IPLookup
from multiprocessing import Pool
from psutil import cpu_count

class FeatureExtracter:
    PACKETS = None
    __winSize = None
    __winType = None
    __featureNames = None
    __minPackets = None
    __featureData = None
    __burstThreshold = None
    __printDetails = False
    __burstCols = ['Type', 'Start Time', 'End Time', 'Packets', 'Bytes']
    __iplookup = None
    __TLSTCPList = ['TLSv1.2', 'TLSv1', 'TLS', 'TLSv1.1', 'TCP']
    __DNSList = ['DNS']
    __UDPList = ['UDP']
    __NTPList = ['NTP']
    def __init__(self, win_size='300s', min_packets=1, win_type='simple', burst_threshold=2):
        self.__winSize = win_size
        self.__minPackets = min_packets
        self.__featureNames = getFeatureNames()
        self.__winType = win_type
        self.__burstThreshold = burst_threshold
        IOTBASE = os.getenv('IOTBASE')
        if IOTBASE is None:
            raise ValueError("Environment Variable 'IOTBASE' not set")
        iplookfile = os.path.join(IOTBASE, 'extras', 'hostnameMap.json')
        self.__iplookup = IPLookup(lookup_file=iplookfile)

# N Count Dicts
    def extractDevicePortCount(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            Device Port Dict
            Device Port TLSTCP
            Device Port DNS
            Device Port UDP
            Device Port NTP
        '''
        device_port_dict = self.extractDevicePortCountHelper(outgoingPackets, incomingPackets, protocols=None)
        tlstcp_device_port = self.extractDevicePortCountHelper(outgoingPackets, incomingPackets, protocols=self.__TLSTCPList)
        dns_device_port = self.extractDevicePortCountHelper(outgoingPackets, incomingPackets, protocols=self.__DNSList)
        udp_device_port = self.extractDevicePortCountHelper(outgoingPackets, incomingPackets, protocols=self.__UDPList)
        ntp_device_port = self.extractDevicePortCountHelper(outgoingPackets, incomingPackets, protocols=self.__NTPList)

        return [
            device_port_dict,
            tlstcp_device_port,
            dns_device_port,
            udp_device_port,
            ntp_device_port]
    def extractExternalPortCount(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            External Port Dict
            External Port TLSTCP
            External Port DNS
            External Port UDP
            External Port NTP
        '''
        external_port_dict = self.extractExternalPortCountHelper(outgoingPackets, incomingPackets, protocols=None)
        tlstcp_external_port = self.extractExternalPortCountHelper(outgoingPackets, incomingPackets, protocols=self.__TLSTCPList)
        dns_external_port = self.extractExternalPortCountHelper(outgoingPackets, incomingPackets, protocols=self.__DNSList)
        udp_external_port = self.extractExternalPortCountHelper(outgoingPackets, incomingPackets, protocols=self.__UDPList)
        ntp_external_port = self.extractExternalPortCountHelper(outgoingPackets, incomingPackets, protocols=self.__NTPList)

        return [
            external_port_dict,
            tlstcp_external_port,
            dns_external_port,
            udp_external_port,
            ntp_external_port]
    def extractContactedIP(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            IP Dict
            IP TLSTCP
            IP DNS
            IP UDP
            IP NTP
        '''
        ip_dict = self.extractContactedIPHelper(allPackets, protocols=None)
        tlstcp_ip = self.extractContactedIPHelper(allPackets, protocols=self.__TLSTCPList)
        dns_ip = self.extractContactedIPHelper(allPackets, protocols=self.__DNSList)
        udp_ip = self.extractContactedIPHelper(allPackets, protocols=self.__UDPList)
        ntp_ip = self.extractContactedIPHelper(allPackets, protocols=self.__NTPList)

        return [
            ip_dict,
            tlstcp_ip,
            dns_ip,
            udp_ip,
            ntp_ip]
    def extractContactedHostName(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            Hostname Dict
            Hostname TLSTCP
            Hostname DNS
            Hostname UDP
            Hostname NTP
        '''
        hostname_dict = self.extractContactedHostNameHelper(allPackets, protocols=None)
        tlstcp_hostname = self.extractContactedHostNameHelper(allPackets, protocols=self.__TLSTCPList)
        dns_hostname = self.extractContactedHostNameHelper(allPackets, protocols=self.__DNSList)
        udp_hostname = self.extractContactedHostNameHelper(allPackets, protocols=self.__UDPList)
        ntp_hostname = self.extractContactedHostNameHelper(allPackets, protocols=self.__NTPList)

        return [
            hostname_dict,
            tlstcp_hostname,
            dns_hostname,
            udp_hostname,
            ntp_hostname]
    def extractProtocols(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            protocols    
        '''
        try:
            protocols = outgoingPackets['Protocol']
        except Exception as e:
            print(e)
            outgoing_srcports = []
            
        return [dict(protocols.value_counts())]
    def extractPacketSizes(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            TLS TCP Packet Sizes
            DNS Packet Sizes
            UDP Packet Sizes
            NTP Packet Sizes
        '''

        tlstcp_packet_sizes = self.extractPacketSizesHelper(allPackets, protocols=['TLSv1.2', 'TLSv1', 'TLS', 'TLSv1.2', 'TCP'])
        dns_packet_sizes = self.extractPacketSizesHelper(allPackets, protocols=['DNS'])
        udp_packet_sizes = self.extractPacketSizesHelper(allPackets, protocols=['UDP'])
        ntp_packet_sizes = self.extractPacketSizesHelper(allPackets, protocols=['NTP'])

        return [
            tlstcp_packet_sizes,
            dns_packet_sizes,
            udp_packet_sizes,
            ntp_packet_sizes]
    def extractPingPongPairs(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            Ping Pong Dict
            Ping Pong TLSTCP
            Ping Pong DNS
            Ping Pong UDP
            Ping Pong NTP
        '''
        pingpong_dict = self.extractPingPongPairsHelper(allPackets, protocols=None)
        pingpong_tlstcp = self.extractPingPongPairsHelper(allPackets, protocols=self.__TLSTCPList)
        pingpong_dns = self.extractPingPongPairsHelper(allPackets, protocols=self.__DNSList)
        pingpong_udp = self.extractPingPongPairsHelper(allPackets, protocols=self.__UDPList)
        pingpong_ntp = self.extractPingPongPairsHelper(allPackets, protocols=self.__NTPList)

        return [
            pingpong_dict,
            pingpong_tlstcp,
            pingpong_dns,
            pingpong_udp,
            pingpong_ntp]

# Helpers
    def groupBurstPackets(self, allPackets):
        if self.__printDetails: print('Burst Threshold:', self.__burstThreshold)
        lstpktip = ''
        lstpkttype = ''
        burstpkts = 0
        burstsize = 0
        startpktime = None
        lstpkttime = None
        newburst = True
        burstsArray=[]
        for pkt in allPackets.iterrows():
            pktType = pkt[1]['PacketType']
            pktTime = pkt[1]['TimeStamp']
            pktIP = pkt[1]['IP']
            pktLength = pkt[1]['Length']
            if lstpkttype == pktType and lstpktip == pktIP:
                if newburst:
                    startpktime = lstpkttime
                    newburst = False
                burstpkts += 1
                burstsize += pktLength
            else:
                if burstpkts >= self.__burstThreshold:
                    burstsArray.append([lstpkttype, startpktime, lstpkttime, burstpkts, burstsize])
                newburst = True
                burstpkts = 1
                burstsize = pktLength
            lstpkttype = pktType
            lstpktip = pktIP
            lstpkttime = pktTime
        if burstpkts >= self.__burstThreshold:
            burstsArray.append([lstpkttype, startpktime, lstpkttime, burstpkts, burstsize])
        return pd.DataFrame(burstsArray, columns=self.__burstCols)
    def __convertTimeStamp(self, timestamps, from_unit='nano', to_unit='milli'):
        value = 0
        if from_unit == 'nano':
            value -= 9
        elif from_unit == 'micro':
            value -= 6
        elif from_unit == 'milli':
            value -= 3
        else:
            raise Exception('from_unit not recognized: {}'.format(from_unit))
        if to_unit == 'nano':
            value += 9
        elif to_unit == 'micro':
            value += 6
        elif to_unit == 'milli':
            value += 3
        else:
            raise Exception('to_unit not recognized: {}'.format(to_unit))
        multiplier = 10 ** value
        retVal = np.float64(timestamps) * multiplier
        try:
            len(retVal)
        except:
            retVal = [retVal]
        finally:
            return retVal
    def extractProtocolBasedInterPacketDelayHelper(self, allPackets, outgoingPackets, incomingPackets, protocols, outgoing=True, incoming=True):
        '''
        FEATURES
            out_mean_inter_proto_pkt_delay          
            out_median_inter_proto_pkt_delay          
            out_25per_inter_proto_pkt_delay
            out_75per_inter_proto_pkt_delay
            out_90per_inter_proto_pkt_delay
            out_std_inter_proto_pkt_delay
            out_max_inter_proto_pkt_delay
            out_min_inter_proto_pkt_delay
            in_mean_inter_proto_pkt_delay          
            in_median_inter_proto_pkt_delay          
            in_25per_inter_proto_pkt_delay
            in_75per_inter_proto_pkt_delay
            in_90per_inter_proto_pkt_delay
            in_std_inter_proto_pkt_delay
            in_max_inter_proto_pkt_delay
            in_min_inter_proto_pkt_delay
        '''
        returnArray = []

        if outgoing:
            try:
                proto_outgoingPackets = outgoingPackets[outgoingPackets['Protocol'].isin(protocols)]
                proto_outgoingPackets_time = proto_outgoingPackets['TimeStamp'].values
                proto_outgoingPackets_interpktdelay = proto_outgoingPackets_time[1:] - proto_outgoingPackets_time[:-1]
                proto_outgoingPackets_interpktdelay = self.__convertTimeStamp(proto_outgoingPackets_interpktdelay)
            except Exception as e:
                print(e)
                proto_outgoingPackets_interpktdelay = []
            if len(proto_outgoingPackets_interpktdelay):
                out_mean_inter_proto_pkt_delay = np.mean(proto_outgoingPackets_interpktdelay)      
                out_median_inter_proto_pkt_delay = np.median(proto_outgoingPackets_interpktdelay)           
                out_25per_inter_proto_pkt_delay = np.percentile(proto_outgoingPackets_interpktdelay, 25)
                out_75per_inter_proto_pkt_delay = np.percentile(proto_outgoingPackets_interpktdelay, 75)
                out_90per_inter_proto_pkt_delay = np.percentile(proto_outgoingPackets_interpktdelay, 90)
                out_std_inter_proto_pkt_delay = np.std(proto_outgoingPackets_interpktdelay)
                out_max_inter_proto_pkt_delay = np.max(proto_outgoingPackets_interpktdelay)
                out_min_inter_proto_pkt_delay = np.min(proto_outgoingPackets_interpktdelay)
            else:
                out_mean_inter_proto_pkt_delay = np.nan           
                out_median_inter_proto_pkt_delay = np.nan           
                out_25per_inter_proto_pkt_delay = np.nan 
                out_75per_inter_proto_pkt_delay = np.nan 
                out_90per_inter_proto_pkt_delay = np.nan 
                out_std_inter_proto_pkt_delay = np.nan 
                out_max_inter_proto_pkt_delay = np.nan 
                out_min_inter_proto_pkt_delay = np.nan 

            returnArray.extend([
                out_mean_inter_proto_pkt_delay,          
                out_median_inter_proto_pkt_delay,          
                out_25per_inter_proto_pkt_delay,
                out_75per_inter_proto_pkt_delay,
                out_90per_inter_proto_pkt_delay,
                out_std_inter_proto_pkt_delay,
                out_max_inter_proto_pkt_delay,
                out_min_inter_proto_pkt_delay
            ])
        if incoming:
            try:
                proto_incomingPackets = incomingPackets[incomingPackets['Protocol'].isin(protocols)]
                proto_incomingPackets_time = proto_incomingPackets['Time'].values
                proto_incomingPackets_interpktdelay = proto_incomingPackets_time[1:] - proto_incomingPackets_time[:-1]
                proto_incomingPackets_interpktdelay = self.__convertTimeStamp(proto_incomingPackets_interpktdelay)
            except Exception as e:
                print(e)
                proto_incomingPackets_interpktdelay = []
        
            if len(proto_incomingPackets_interpktdelay):
                in_mean_inter_proto_pkt_delay = np.mean(proto_incomingPackets_interpktdelay)      
                in_median_inter_proto_pkt_delay = np.median(proto_incomingPackets_interpktdelay)           
                in_25per_inter_proto_pkt_delay = np.percentile(proto_incomingPackets_interpktdelay, 25)
                in_75per_inter_proto_pkt_delay = np.percentile(proto_incomingPackets_interpktdelay, 75)
                in_90per_inter_proto_pkt_delay = np.percentile(proto_incomingPackets_interpktdelay, 90)
                in_std_inter_proto_pkt_delay = np.std(proto_incomingPackets_interpktdelay)
                in_max_inter_proto_pkt_delay = np.max(proto_incomingPackets_interpktdelay)
                in_min_inter_proto_pkt_delay = np.min(proto_incomingPackets_interpktdelay)
            else:
                in_mean_inter_proto_pkt_delay = np.nan           
                in_median_inter_proto_pkt_delay = np.nan           
                in_25per_inter_proto_pkt_delay = np.nan 
                in_75per_inter_proto_pkt_delay = np.nan 
                in_90per_inter_proto_pkt_delay = np.nan 
                in_std_inter_proto_pkt_delay = np.nan 
                in_max_inter_proto_pkt_delay = np.nan 
                in_min_inter_proto_pkt_delay = np.nan 

            returnArray.extend([
                in_mean_inter_proto_pkt_delay,          
                in_median_inter_proto_pkt_delay,          
                in_25per_inter_proto_pkt_delay,
                in_75per_inter_proto_pkt_delay,
                in_90per_inter_proto_pkt_delay,
                in_std_inter_proto_pkt_delay,
                in_max_inter_proto_pkt_delay,
                in_min_inter_proto_pkt_delay
            ])
        return returnArray
    def extractDevicePortCountHelper(self, outgoingPackets, incomingPackets, protocols):
        '''
        FEATURES
            device_ports
        '''
        try:
            if protocols:
                outgoing_protocol_packets = outgoingPackets['Protocol'].isin(protocols)
                outgoing_srcports = outgoingPackets.loc[outgoing_protocol_packets, 'SrcPort'].values
            else:
                outgoing_srcports = outgoingPackets['SrcPort'].values
        except Exception as e:
            print('ERROR | extractDevicePortCountHelper', e)
            outgoing_srcports = []
        
        try:
            if protocols:
                incoming_protocol_packets = incomingPackets['Protocol'].isin(protocols)
                incoming_dstports = incomingPackets.loc[incoming_protocol_packets, 'DstPort'].values
            else:
                incoming_dstports = incomingPackets['DstPort'].values
        except Exception as e:
            print('ERROR | extractDevicePortCountHelper', e)
            incoming_dstports = []

        device_ports = pd.Series(np.hstack([outgoing_srcports, incoming_dstports]))

        return dict(device_ports.value_counts())
    def extractExternalPortCountHelper(self, outgoingPackets, incomingPackets, protocols):
        '''
        FEATURES
            external_ports
        '''
        try:
            if protocols:
                outgoing_protocol_packets = outgoingPackets['Protocol'].isin(protocols)
                outgoing_dstports = outgoingPackets.loc[outgoing_protocol_packets, 'DstPort'].values
            else:
                outgoing_dstports = outgoingPackets['DstPort'].values
        except Exception as e:
            print('ERROR | extractExternalPortCountHelper', e)
            outgoing_dstports = []
        
        try:
            if protocols:
                incoming_protocol_packets = incomingPackets['Protocol'].isin(protocols)
                incoming_srcports = incomingPackets.loc[incoming_protocol_packets, 'SrcPort'].values
            else:
                incoming_srcports = incomingPackets['SrcPort'].values
        except Exception as e:
            print('ERROR | extractExternalPortCountHelper', e)
            incoming_srcports = []

        external_ports = pd.Series(np.hstack([outgoing_dstports, incoming_srcports]))

        return dict(external_ports.value_counts())
    def extractContactedIPHelper(self, allPackets, protocols):
        '''
        FEATURES
        contacted_ip
        '''
        try:
            if protocols:
                proto_allPackets = allPackets['Protocol'].isin(protocols)
                contacted_ips = allPackets.loc[proto_allPackets, 'IP']
            else:
                contacted_ips = allPackets['IP']
        except Exception as e:
            print('ERROR | extractContactedIPHelper:', e)
            contacted_ips = pd.DataFrame([])
        try:
            contacted_ip = contacted_ips.apply(self.__iplookup.lookupIP)
        except Exception as e: 
            print('ERROR | extractContactedIPHelper:',e)
            contacted_ip = pd.Series([])
        
        return dict(contacted_ip.value_counts())
    def extractContactedHostNameHelper(self, allPackets, protocols):
        '''
        FEATURES
        contacted_hostname
        '''
        try:
            if protocols:
                proto_allPackets = allPackets['Protocol'].isin(protocols)
                contacted_ips = allPackets.loc[proto_allPackets, 'IP']
            else:
                contacted_ips = allPackets['IP']
        except Exception as e:
            print('ERROR | extractContactedHostNameHelper:', e)
            contacted_ips = pd.DataFrame([])
        try:
            contacted_hostname = contacted_ips.apply(self.__iplookup.lookupHostName)
        except Exception as e: 
            print('Error:',e)
            contacted_hostname = pd.Series([])
        
        return dict(contacted_hostname.value_counts())
    def extractPacketSizesHelper(self, allPackets, protocols):
        '''
        FEATURES
        packet_sizes
        '''
        try:
            proto_allPackets = allPackets[allPackets['Protocol'].isin(protocols)]
            packet_sizes = proto_allPackets['Length']
        except Exception as e: 
            print('Error:',e)
            packet_sizes = pd.Series([])
        
        return dict(packet_sizes.value_counts())
    def extractPingPongPairsHelper(self, allPackets, protocols):
        '''
        FEATURES
            Ping Pong
        '''
        if protocols:
            packets = allPackets[allPackets['Protocol'].isin(protocols)]
        else:
            packets = allPackets
        pingPongPairs = pd.DataFrame([], columns=['PingPong'])
        i = 0
        keyValStore = {} 
        for _, pkt in packets.iterrows():
            if pkt['PacketType'] == 'Outgoing':
                dev_port = pkt['SrcPort']
                ext_port = pkt['DstPort']
            elif pkt['PacketType'] == 'Incoming':
                dev_port = pkt['DstPort']
                ext_port = pkt['SrcPort']
            else:
                raise AssertionError(f"Unknown PacketType {pkt['PacketType']}")
            conn_id = (pkt['IP'], dev_port, ext_port, pkt['Protocol']).__hash__()
            if pkt['PacketType'] == 'Outgoing':
                keyValStore[conn_id] = pkt['Length']
            elif pkt['PacketType'] == 'Incoming':
                if conn_id in keyValStore:
                    pingPongPairs.loc[i, 'PingPong'] = (keyValStore[conn_id], pkt['Length'])
                    i += 1
                    del keyValStore[conn_id]
            else:
                raise AssertionError(f"Unknown PacketType {pkt['PacketType']}")
        return dict(pingPongPairs['PingPong'].value_counts())
# Feature Extracting Functions Level 2
    def __extractTotalPkts(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES:
            out_totalpkts
            in_totalpkts
        '''
        try:
            out_totalpkts = outgoingPackets.shape[0]
        except Exception as e:
            print('__extractTotalPkts', e)
            out_totalpkts = np.nan
        try:
            in_totalpkts = incomingPackets.shape[0]
        except Exception as e:
            print('__extractTotalPkts', e)
            in_totalpkts = np.nan
        return [out_totalpkts, in_totalpkts]
    def __extractTotalBytes(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES: 
            out_totalbytes
            in_totalbytes
        '''
        try:
            out_totalbytes = outgoingPackets['Length'].sum()
        except Exception as e:
            print('__extractTotalBytes', e)
            out_totalbytes = np.nan
        try:
            in_totalbytes = incomingPackets['Length'].sum()
        except Exception as e:
            print('__extractTotalBytes', e)
            in_totalbytes = np.nan
        return [
            out_totalbytes,
            in_totalbytes
        ]
    def __extractUniqueLen(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES: 
            mean_out_uniquelen
            mean_in_uniquelen
            median_out_uniquelen
            median_in_uniquelen
            25per_out_uniquelen
            25per_in_uniquelen
            75per_out_uniquelen
            75per_in_uniquelen
            90per_out_uniquelen
            90per_in_uniquelen
            len_out_uniquelen
            len_in_uniquelen
            max_out_uniquelen
            max_in_uniquelen
            min_out_uniquelen
            min_in_uniquelen
        '''
        try:
            outgoing_uniqueLen = outgoingPackets['Length'].unique()
        except Exception as e:
            print('__extractUniqueLen', e)
            outgoing_uniqueLen = []
        try:
            incoming_uniqueLen = incomingPackets['Length'].unique()
        except Exception as e:
            print('__extractUniqueLen', e)
            incoming_uniqueLen = []
        
        try:
            if len(outgoing_uniqueLen) > 0:
                mean_out_uniquelen = np.mean(outgoing_uniqueLen)
                median_out_uniquelen = np.median(outgoing_uniqueLen)
                out_25per_uniquelen = np.percentile(outgoing_uniqueLen, 25)
                out_75per_uniquelen = np.percentile(outgoing_uniqueLen, 75)
                out_90per_uniquelen = np.percentile(outgoing_uniqueLen, 90)
                len_out_uniquelen = len(outgoing_uniqueLen)
                max_out_uniquelen = np.max(outgoing_uniqueLen)
                min_out_uniquelen = np.min(outgoing_uniqueLen)
            else:
                mean_out_uniquelen = np.nan
                median_out_uniquelen = np.nan
                out_25per_uniquelen = np.nan
                out_75per_uniquelen = np.nan
                out_90per_uniquelen = np.nan 
                len_out_uniquelen = np.nan              
                max_out_uniquelen = np.nan
                min_out_uniquelen = np.nan
        except Exception as e:
            print('__extractUniqueLen', e)
            raise(e)

        try:
            if len(incoming_uniqueLen) > 0:
                mean_in_uniquelen = np.mean(incoming_uniqueLen)
                median_in_uniquelen = np.median(incoming_uniqueLen)
                in_25per_uniquelen = np.percentile(incoming_uniqueLen, 25)
                in_75per_uniquelen = np.percentile(incoming_uniqueLen, 75)
                in_90per_uniquelen = np.percentile(incoming_uniqueLen, 90)
                len_in_uniquelen = len(incoming_uniqueLen)
                max_in_uniquelen = np.max(incoming_uniqueLen)
                min_in_uniquelen = np.min(incoming_uniqueLen)
            else:
                mean_in_uniquelen = np.nan
                median_in_uniquelen = np.nan
                in_25per_uniquelen = np.nan
                in_75per_uniquelen = np.nan
                in_90per_uniquelen = np.nan
                len_in_uniquelen = np.nan
                max_in_uniquelen = np.nan
                min_in_uniquelen = np.nan
        except Exception as e:
            print('__extractUniqueLen', e)
            raise(e)

        return [
            mean_out_uniquelen,
            mean_in_uniquelen,
            median_out_uniquelen,
            median_in_uniquelen,
            out_25per_uniquelen,
            in_25per_uniquelen,
            out_75per_uniquelen,
            in_75per_uniquelen,
            out_90per_uniquelen,
            in_90per_uniquelen,
            len_out_uniquelen,
            len_in_uniquelen,
            max_out_uniquelen,
            max_in_uniquelen,
            min_out_uniquelen,
            min_in_uniquelen
        ]
    def __extractPacketPercentage(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES: 
            out_percentage
            in_percentage
        '''
        try:
            out_percentage = float(outgoingPackets.shape[0]) / float(allPackets.shape[0])
        except:
            out_percentage = np.nan
        try:
            in_percentage = float(incomingPackets.shape[0]) / float(allPackets.shape[0])
        except:
            in_percentage = np.nan

        return [
            out_percentage,
            in_percentage
        ]
    def __extractTCPFlags(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES: 
            out_tcpack_percentage
            out_tcpsyn_percentage
            out_tcpfin_percentage
            out_tcprst_percentage
            out_tcppsh_percentage
            out_tcpurg_percentage
            in_tcpack_percentage
            in_tcpsyn_percentage
            in_tcpfin_percentage
            in_tcprst_percentage
            in_tcppsh_percentage
            in_tcpurg_percentage

        '''
        
        try:
            out_ackflagSum = outgoingPackets['tcpACK'].sum()
            out_synflagSum = outgoingPackets['tcpSYN'].sum()
            out_finflagSum = outgoingPackets['tcpFIN'].sum()
            out_rstflagSum = outgoingPackets['tcpRST'].sum()
            out_pshflagSum = outgoingPackets['tcpPSH'].sum()
            out_urgflagSum = outgoingPackets['tcpURG'].sum()
            
            totalSum = out_ackflagSum + out_synflagSum + out_finflagSum + out_rstflagSum + out_pshflagSum + out_urgflagSum
        
            if totalSum > 0:
                out_tcpack_percentage = out_ackflagSum / totalSum
                out_tcpsyn_percentage = out_synflagSum / totalSum
                out_tcpfin_percentage = out_finflagSum / totalSum
                out_tcprst_percentage = out_rstflagSum / totalSum
                out_tcppsh_percentage = out_pshflagSum / totalSum
                out_tcpurg_percentage = out_urgflagSum / totalSum
            else:
                out_tcpack_percentage = np.nan
                out_tcpsyn_percentage = np.nan
                out_tcpfin_percentage = np.nan
                out_tcprst_percentage = np.nan
                out_tcppsh_percentage = np.nan
                out_tcpurg_percentage = np.nan

        except Exception as e:
            print(e)
            out_tcpack_percentage = np.nan
            out_tcpsyn_percentage = np.nan
            out_tcpfin_percentage = np.nan
            out_tcprst_percentage = np.nan
            out_tcppsh_percentage = np.nan
            out_tcpurg_percentage = np.nan
        
        
        try:
            in_ackflagSum = outgoingPackets['tcpACK'].sum()
            in_synflagSum = outgoingPackets['tcpSYN'].sum()
            in_finflagSum = outgoingPackets['tcpFIN'].sum()
            in_rstflagSum = outgoingPackets['tcpRST'].sum()
            in_pshflagSum = outgoingPackets['tcpPSH'].sum()
            in_urgflagSum = outgoingPackets['tcpURG'].sum()
            
            totalSum = in_ackflagSum + in_synflagSum + in_finflagSum + in_rstflagSum + in_pshflagSum + in_urgflagSum
            
            if totalSum > 0:
                in_tcpack_percentage = in_ackflagSum / totalSum
                in_tcpsyn_percentage = in_synflagSum / totalSum
                in_tcpfin_percentage = in_finflagSum / totalSum
                in_tcprst_percentage = in_rstflagSum / totalSum
                in_tcppsh_percentage = in_pshflagSum / totalSum
                in_tcpurg_percentage = in_urgflagSum / totalSum
            else:
                in_tcpack_percentage = np.nan
                in_tcpsyn_percentage = np.nan
                in_tcpfin_percentage = np.nan
                in_tcprst_percentage = np.nan
                in_tcppsh_percentage = np.nan
                in_tcpurg_percentage = np.nan
        except Exception as e:
            print(e)
            in_tcpack_percentage = np.nan
            in_tcpsyn_percentage = np.nan
            in_tcpfin_percentage = np.nan
            in_tcprst_percentage = np.nan
            in_tcppsh_percentage = np.nan
            in_tcpurg_percentage = np.nan
        
        return [
            out_tcpack_percentage,
            out_tcpsyn_percentage,
            out_tcpfin_percentage,
            out_tcprst_percentage,
            out_tcppsh_percentage,
            out_tcpurg_percentage,
            in_tcpack_percentage,
            in_tcpsyn_percentage,
            in_tcpfin_percentage,
            in_tcprst_percentage,
            in_tcppsh_percentage,
            in_tcpurg_percentage]
    def __extractProtocols(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            out_tls1pkts_percentage
            in_tls1pkts_percentage
            out_tls12pkts_percentage
            in_tls12pkts_percentage
            out_tcppkts_percentage
            in_tcppkts_percentage
            out_udppkts_percentage
            in_udppkts_percentage
            out_dnspkts_percentage
            in_dnspkts_percentage
            out_ssdppkts_percentage
            in_ssdppkts_percentage
            out_sslpkts_percentage
            in_sslpkts_percentage
            out_icmppkts_percentage
            in_icmppkts_percentage
            out_ntppkts_percentage
            in_ntppkts_percentage
        '''
        out_tls1pkts_percentage = np.nan
        in_tls1pkts_percentage = np.nan
        out_tls12pkts_percentage = np.nan
        in_tls12pkts_percentage = np.nan
        out_tcppkts_percentage = np.nan
        in_tcppkts_percentage = np.nan
        out_udppkts_percentage = np.nan
        in_udppkts_percentage = np.nan
        out_dnspkts_percentage = np.nan
        in_dnspkts_percentage = np.nan
        out_ssdppkts_percentage = np.nan
        in_ssdppkts_percentage = np.nan
        out_sslpkts_percentage = np.nan
        in_sslpkts_percentage = np.nan
        out_icmppkts_percentage = np.nan
        in_icmppkts_percentage = np.nan
        out_ntppkts_percentage = np.nan
        in_ntppkts_percentage = np.nan

        totalOutgoingPackets = outgoingPackets.shape[0]
        totalIncomingPackets = incomingPackets.shape[0]

        if isinstance(outgoingPackets['Protocol'], pd.Series) and totalOutgoingPackets > 0:
            out_tls1pkts_percentage = (outgoingPackets['Protocol'] == 'TLSv1').sum() / totalOutgoingPackets
            out_tls12pkts_percentage = (outgoingPackets['Protocol'] == 'TLSv1.2').sum() / totalOutgoingPackets
            out_tcppkts_percentage = (outgoingPackets['Protocol'] == 'TCP').sum() / totalOutgoingPackets
            out_udppkts_percentage = (outgoingPackets['Protocol'] == 'UDP').sum() / totalOutgoingPackets
            out_dnspkts_percentage = (outgoingPackets['Protocol'] == 'DNS').sum() / totalOutgoingPackets
            out_ssdppkts_percentage = (outgoingPackets['Protocol'] == 'SSDP').sum() / totalOutgoingPackets
            out_sslpkts_percentage = (outgoingPackets['Protocol'] == 'SSL').sum() / totalOutgoingPackets
            out_icmppkts_percentage = (outgoingPackets['Protocol'] == 'ICMP').sum() / totalOutgoingPackets
            out_ntppkts_percentage = (outgoingPackets['Protocol'] == 'NTP').sum() / totalOutgoingPackets
        else:
            out_tls1pkts_percentage = np.nan
            out_tls12pkts_percentage = np.nan
            out_tcppkts_percentage= np.nan
            out_udppkts_percentage= np.nan
            out_dnspkts_percentage = np.nan
            out_ssdppkts_percentage = np.nan
            out_sslpkts_percentage = np.nan
            out_icmppkts_percentage = np.nan
            out_ntppkts_percentage = np.nan

        if isinstance(incomingPackets['Protocol'], pd.Series) and totalIncomingPackets > 0:
            in_tls1pkts_percentage = (incomingPackets['Protocol'] == 'TLSv1').sum() / totalIncomingPackets
            in_tls12pkts_percentage = (incomingPackets['Protocol'] == 'TLSv1.2').sum() / totalIncomingPackets
            in_tcppkts_percentage = (incomingPackets['Protocol'] == 'TCP').sum() / totalIncomingPackets
            in_udppkts_percentage = (incomingPackets['Protocol'] == 'UDP').sum() / totalIncomingPackets
            in_dnspkts_percentage = (incomingPackets['Protocol'] == 'DNS').sum() / totalIncomingPackets
            in_ssdppkts_percentage = (incomingPackets['Protocol'] == 'SSDP').sum() / totalIncomingPackets
            in_sslpkts_percentage = (incomingPackets['Protocol'] == 'SSL').sum() / totalIncomingPackets
            in_icmppkts_percentage = (incomingPackets['Protocol'] == 'ICMP').sum() / totalIncomingPackets
            in_ntppkts_percentage = (incomingPackets['Protocol'] == 'NTP').sum() / totalIncomingPackets
        else:
            in_tls1pkts_percentage = np.nan
            in_tls12pkts_percentage = np.nan
            in_tcppkts_percentage = np.nan
            in_udppkts_percentage = np.nan
            in_dnspkts_percentage = np.nan
            in_ssdppkts_percentage = np.nan
            in_sslpkts_percentage = np.nan
            in_icmppkts_percentage = np.nan
            in_ntppkts_percentage = np.nan

        return [
            out_tls1pkts_percentage,
            in_tls1pkts_percentage,
            out_tls12pkts_percentage,
            in_tls12pkts_percentage,
            out_tcppkts_percentage,
            in_tcppkts_percentage,
            out_udppkts_percentage,
            in_udppkts_percentage,
            out_dnspkts_percentage,
            in_dnspkts_percentage,
            out_ssdppkts_percentage,
            in_ssdppkts_percentage,
            out_sslpkts_percentage,
            in_sslpkts_percentage,
            out_icmppkts_percentage,
            in_icmppkts_percentage,
            out_ntppkts_percentage,
            in_ntppkts_percentage
        ]
    def __extractUniqueProtocols(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            out_numuniqueprotocol
            in_numuniqueprotocol
        '''
        try:
            out_numuniqueprotocol = outgoingPackets['Protocol'].nunique()
        except:
            out_numuniqueprotocol = 0
        try:
            in_numuniqueprotocol = incomingPackets['Protocol'].nunique()
        except:
            in_numuniqueprotocol = 0

        return [
            out_numuniqueprotocol,
            in_numuniqueprotocol
        ]
    def __extractHostNameIP(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            num_unique_ip
            num_unique_ip_3octet
            num_unique_hostname
        '''
        try:
            num_unique_ip = allPackets['IP'].nunique()
        except Exception as e:
            print(e)
            num_unique_ip = np.nan
        try:
            num_unique_ip_3octet = allPackets['IP'].apply(self.__iplookup.lookupIP).nunique()
        except Exception as e:
            print(e)
            num_unique_ip_3octet = np.nan
        try:
            num_unique_hostname = allPackets['IP'].apply(self.__iplookup.lookupHostName).nunique()
        except Exception as e:
            print(e)
            num_unique_hostname = np.nan
        return [
            num_unique_ip,
            num_unique_ip_3octet,
            num_unique_hostname
        ]
    def __extractUniqueSrcDstPorts(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            out_numuniquesrcport
            in_numuniquesrcport
            out_numuniquedstport
            in_numuniquedstport
        '''

        try:
            out_numuniquesrcport = outgoingPackets['SrcPort'].nunique()
        except Exception as e:
            print(e)
            out_numuniquesrcport = np.nan
        try:
            in_numuniquesrcport = incomingPackets['SrcPort'].nunique()
        except Exception as e:
            print(e)
            in_numuniquesrcport = np.nan

        try:
            out_numuniquedstport = outgoingPackets['DstPort'].nunique()
        except Exception as e:
            print(e)
            out_numuniquedstport = np.nan
        try:    
            in_numuniquedstport = incomingPackets['DstPort'].nunique()
        except Exception as e:
            print(e)
            in_numuniquedstport = np.nan
            
        return [
            out_numuniquesrcport,
            in_numuniquesrcport,
            out_numuniquedstport,
            in_numuniquedstport
        ]
    def __extract80and443Features(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
        'pkts_80_443_percentage'
        'byte_per_pkt_80_443'
        '''
        try:
            outTraffic = outgoingPackets['DstPort'].isin(['80', '443'])
        except Exception as e:
            print('__extract80and443Features', e)
            raise(e)
        try:
            inTraffic = incomingPackets['SrcPort'].isin(['80','443'])
        except Exception as e:
            print('__extract80and443Features', e)
            raise(e)

        totalOutTrafficBytes = outgoingPackets.loc[outTraffic, 'Length'].sum()        
        totalOutTrafficPkts = outgoingPackets[outTraffic].shape[0]
        
        totalInTrafficBytes = incomingPackets.loc[inTraffic, 'Length'].sum()        
        totalInTrafficPkts = incomingPackets[inTraffic].shape[0]

        totalPkts = allPackets.shape[0]
        totalTrafficPkts = totalOutTrafficPkts + totalInTrafficPkts

        if totalPkts > 0:
            pkts_80_443_percentage = (totalOutTrafficPkts + totalInTrafficPkts) / (allPackets.shape[0])
        else:
            pkts_80_443_percentage = np.nan
        
        if totalTrafficPkts > 0:
            byte_per_pkt_80_443 = (totalOutTrafficBytes + totalInTrafficBytes) / (totalOutTrafficPkts + totalInTrafficPkts)
        else:
            byte_per_pkt_80_443 = np.nan
        
        return [
            pkts_80_443_percentage,
            byte_per_pkt_80_443 ]
    def __extractInterPacketDelay(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
            mean_interpktdelay          
            median_interpktdelay          
            25per_interpktdelay
            75per_interpktdelay
            90per_interpktdelay
            std_interpktdelay
            max_interpktdelay
            min_interpktdelay
        '''
        try:
            allPacketsTimeStamps = allPackets['TimeStamp'].values
            interpktdelays = allPacketsTimeStamps[1:] - allPacketsTimeStamps[:-1]
            interpktdelays = self.__convertTimeStamp(interpktdelays)
        except Exception as e:
            print('__extractInterPacketDelay', e)
            interpktdelays = []
        
        if len(interpktdelays) > 0:
            mean_interpktdelay = np.mean(interpktdelays)
            median_interpktdelay = np.median(interpktdelays)
            per25_interpktdelay = np.percentile(interpktdelays, 25)
            per75_interpktdelay = np.percentile(interpktdelays, 75)
            per90_interpktdelay = np.percentile(interpktdelays, 90)
            std_interpktdelay = np.std(interpktdelays)
            max_interpktdelay = np.max(interpktdelays)
            min_interpktdelay = np.min(interpktdelays)
        else:
            mean_interpktdelay = np.nan
            median_interpktdelay = np.nan
            per25_interpktdelay = np.nan
            per75_interpktdelay = np.nan
            per90_interpktdelay = np.nan
            std_interpktdelay = np.nan
            max_interpktdelay = np.nan
            min_interpktdelay = np.nan

        return [
            mean_interpktdelay, 
            median_interpktdelay,          
            per25_interpktdelay,
            per75_interpktdelay,
            per90_interpktdelay,
            std_interpktdelay,
            max_interpktdelay,
            min_interpktdelay
        ]
    def __extractProtocolBasedInterPacketDelay(self, allPackets, outgoingPackets, incomingPackets):
        '''
        FEATURES
        TLS IN/OUT 
        TCP IN/OUT
        DNS IN/OUT
        UDP IN/OUT
        NTP IN/OUT
        '''
        returnArray = []
        
        tls_in_out = self.extractProtocolBasedInterPacketDelayHelper(allPackets, outgoingPackets, incomingPackets, protocols=['TLSv1', 'TLSv1.2'], outgoing=True, incoming=True)
        tcp_in_out = self.extractProtocolBasedInterPacketDelayHelper(allPackets, outgoingPackets, incomingPackets, protocols=['TCP'], outgoing=True, incoming=True)
        dns_in_out = self.extractProtocolBasedInterPacketDelayHelper(allPackets, outgoingPackets, incomingPackets, protocols=['DNS'], outgoing=True, incoming=True)
        udp_in_out = self.extractProtocolBasedInterPacketDelayHelper(allPackets, outgoingPackets, incomingPackets, protocols=['UDP'], outgoing=True, incoming=True)
        ntp_in_out = self.extractProtocolBasedInterPacketDelayHelper(allPackets, outgoingPackets, incomingPackets, protocols=['NTP'], outgoing=True, incoming=True)

        returnArray.extend(tls_in_out)
        returnArray.extend(tcp_in_out)
        returnArray.extend(dns_in_out)
        returnArray.extend(udp_in_out)
        returnArray.extend(ntp_in_out)

        return returnArray
    def __extractInterBurstDelay(self, allBursts, outgoingBursts, incomingBursts):
        '''
        FEATURES
            out_mean_interburstdelay 
            in_mean_interburstdelay
            out_median_interburstdelay
            in_median_interburstdelay 
            out_25per_interburstdelay
            in_25per_interburstdelay
            out_75per_interburstdelay
            in_75per_interburstdelay
            out_90per_interburstdelay
            in_90per_interburstdelay
            out_std_interburstdelay
            in_std_interburstdelay
            out_max_interburstdelay 
            in_max_interburstdelay
            out_min_interburstdelay 
            in_min_interburstdelay
        '''
        try:
            out_startTime = outgoingBursts['Start Time'].values[1:]
            out_endTime = outgoingBursts['End Time'].values[:-1]
            out_interburstdelays = out_startTime - out_endTime
            out_interburstdelays = self.__convertTimeStamp(out_interburstdelays)
        except Exception as e:
            print(e)
            out_interburstdelays = []
        try:
            in_startTime = incomingBursts['Start Time'].values[1:]
            in_endTime = incomingBursts['End Time'].values[:-1]
            in_interburstdelays = in_startTime - in_endTime
            in_interburstdelays = self.__convertTimeStamp(in_interburstdelays)
        except Exception as e:
            print(e)
            in_interburstdelays = []
        if len(out_interburstdelays):
            out_mean_interburstdelay = np.mean(out_interburstdelays)
            out_median_interburstdelay = np.median(out_interburstdelays)
            out_25per_interburstdelay = np.percentile(out_interburstdelays, 25)
            out_75per_interburstdelay = np.percentile(out_interburstdelays, 75)
            out_90per_interburstdelay = np.percentile(out_interburstdelays, 90)
            out_std_interburstdelay = np.std(out_interburstdelays)
            out_max_interburstdelay = np.max(out_interburstdelays)
            out_min_interburstdelay = np.min(out_interburstdelays)
        else:
            out_mean_interburstdelay = np.nan
            out_median_interburstdelay = np.nan
            out_25per_interburstdelay = np.nan
            out_75per_interburstdelay = np.nan
            out_90per_interburstdelay = np.nan
            out_std_interburstdelay = np.nan
            out_max_interburstdelay = np.nan
            out_min_interburstdelay = np.nan
        if len(in_interburstdelays):
            in_mean_interburstdelay = np.mean(in_interburstdelays)
            in_median_interburstdelay = np.median(in_interburstdelays)
            in_25per_interburstdelay = np.percentile(in_interburstdelays, 25)
            in_75per_interburstdelay = np.percentile(in_interburstdelays, 75)
            in_90per_interburstdelay = np.percentile(in_interburstdelays, 90)
            in_std_interburstdelay = np.std(in_interburstdelays)
            in_max_interburstdelay = np.max(in_interburstdelays)
            in_min_interburstdelay = np.min(in_interburstdelays)
        else:
            in_mean_interburstdelay = np.nan
            in_median_interburstdelay = np.nan
            in_25per_interburstdelay = np.nan
            in_75per_interburstdelay = np.nan
            in_90per_interburstdelay = np.nan
            in_std_interburstdelay = np.nan
            in_max_interburstdelay = np.nan
            in_min_interburstdelay = np.nan
     
        return [
            out_mean_interburstdelay, 
            in_mean_interburstdelay,
            out_median_interburstdelay,
            in_median_interburstdelay, 
            out_25per_interburstdelay,
            in_25per_interburstdelay,
            out_75per_interburstdelay,
            in_75per_interburstdelay,
            out_90per_interburstdelay,
            in_90per_interburstdelay,
            out_std_interburstdelay,
            in_std_interburstdelay,
            out_max_interburstdelay, 
            in_max_interburstdelay,
            out_min_interburstdelay, 
            in_min_interburstdelay
        ]
    def __extractBurstNumPackets(self, allBursts, outgoingBursts, incomingBursts):
        '''
        FEATURES
        out_mean_burstnumpkts
        in_mean_burstnumpkts
        out_median_burstnumpkts
        in_median_burstnumpkts
        out_25per_burstnumpkts
        in_25per_burstnumpkts
        out_75per_burstnumpkts
        in_75per_burstnumpkts
        out_90per_burstnumpkts
        in_90per_burstnumpkts
        out_std_burstnumpkts
        in_std_burstnumpkts
        out_max_burstnumpkts
        in_max_burstnumpkts
        out_min_burstnumpkts
        in_min_burstnumpkts
        '''
        try:
            out_burstnumpkts = outgoingBursts['Packets'].values
        except Exception as e:
            print(e)
            out_burstnumpkts = []
        try:
            in_burstnumpkts = incomingBursts['Packets'].values
        except Exception as e:
            print(e)
            in_burstnumpkts = []

        if len(out_burstnumpkts):
            out_mean_burstnumpkts = np.mean(out_burstnumpkts)
            out_median_burstnumpkts = np.median(out_burstnumpkts)
            out_25per_burstnumpkts = np.percentile(out_burstnumpkts, 25)
            out_75per_burstnumpkts = np.percentile(out_burstnumpkts, 75)
            out_90per_burstnumpkts = np.percentile(out_burstnumpkts, 90)
            out_std_burstnumpkts = np.std(out_burstnumpkts)
            out_max_burstnumpkts = np.max(out_burstnumpkts)
            out_min_burstnumpkts = np.min(out_burstnumpkts)
        else:
            out_mean_burstnumpkts = np.nan 
            out_median_burstnumpkts = np.nan
            out_25per_burstnumpkts = np.nan
            out_75per_burstnumpkts = np.nan
            out_90per_burstnumpkts = np.nan
            out_std_burstnumpkts = np.nan
            out_max_burstnumpkts = np.nan
            out_min_burstnumpkts = np.nan
        
        if len(in_burstnumpkts):
            in_mean_burstnumpkts = np.mean(in_burstnumpkts)
            in_median_burstnumpkts = np.median(in_burstnumpkts)
            in_25per_burstnumpkts = np.percentile(in_burstnumpkts, 25)
            in_75per_burstnumpkts = np.percentile(in_burstnumpkts, 75)
            in_90per_burstnumpkts = np.percentile(in_burstnumpkts, 90)
            in_std_burstnumpkts = np.std(in_burstnumpkts)
            in_max_burstnumpkts = np.max(in_burstnumpkts)
            in_min_burstnumpkts = np.min(in_burstnumpkts)
        else:
            in_mean_burstnumpkts = np.nan
            in_median_burstnumpkts = np.nan
            in_25per_burstnumpkts = np.nan
            in_75per_burstnumpkts = np.nan
            in_90per_burstnumpkts = np.nan
            in_std_burstnumpkts = np.nan
            in_max_burstnumpkts = np.nan
            in_min_burstnumpkts = np.nan

        return [
            out_mean_burstnumpkts,
            in_mean_burstnumpkts,
            out_median_burstnumpkts,
            in_median_burstnumpkts,
            out_25per_burstnumpkts,
            in_25per_burstnumpkts,
            out_75per_burstnumpkts,
            in_75per_burstnumpkts,
            out_90per_burstnumpkts,
            in_90per_burstnumpkts,
            out_std_burstnumpkts,
            in_std_burstnumpkts,
            out_max_burstnumpkts,
            in_max_burstnumpkts,
            out_min_burstnumpkts,
            in_min_burstnumpkts
        ]
    def __extractBurstBytes(self, allBursts, outgoingBursts, incomingBursts):
        '''
        FEATURES
        out_mean_burstbytes
        in_mean_burstbytes
        out_median_burstbytes
        in_median_burstbytes
        out_25per_burstbytes
        in_25per_burstbytes
        out_75per_burstbytes
        in_75per_burstbytes
        out_90per_burstbytes
        in_90per_burstbytes
        out_std_burstbytes
        in_std_burstbytes
        out_max_burstbytes
        in_max_burstbytes
        out_min_burstbytes
        in_min_burstbytes
        '''
        try:
            out_burstbytes = outgoingBursts['Bytes'].values
        except Exception as e:
            print(e)
            out_burstbytes = []
        try:
            in_burstbytes = incomingBursts['Bytes'].values
        except Exception as e:
            print(e)
            in_burstbytes = []
        
        if len(out_burstbytes):
            out_mean_burstbytes = np.mean(out_burstbytes)
            out_median_burstbytes = np.median(out_burstbytes)
            out_25per_burstbytes = np.percentile(out_burstbytes, 25)
            out_75per_burstbytes = np.percentile(out_burstbytes, 75)
            out_90per_burstbytes = np.percentile(out_burstbytes, 90)
            out_std_burstbytes = np.std(out_burstbytes)
            out_max_burstbytes = np.max(out_burstbytes)
            out_min_burstbytes = np.min(out_burstbytes)
        else:
            out_mean_burstbytes = np.nan
            out_median_burstbytes = np.nan
            out_25per_burstbytes = np.nan
            out_75per_burstbytes = np.nan
            out_90per_burstbytes = np.nan
            out_std_burstbytes = np.nan
            out_max_burstbytes = np.nan
            out_min_burstbytes = np.nan
        
        if len(in_burstbytes):
            in_mean_burstbytes = np.mean(in_burstbytes)
            in_median_burstbytes = np.median(in_burstbytes)
            in_25per_burstbytes = np.percentile(in_burstbytes, 25)
            in_75per_burstbytes = np.percentile(in_burstbytes, 75)
            in_90per_burstbytes = np.percentile(in_burstbytes, 90)
            in_std_burstbytes = np.std(in_burstbytes)
            in_max_burstbytes = np.max(in_burstbytes)
            in_min_burstbytes = np.min(in_burstbytes)
        else:
            in_mean_burstbytes = np.nan
            in_median_burstbytes = np.nan
            in_25per_burstbytes = np.nan
            in_75per_burstbytes = np.nan
            in_90per_burstbytes = np.nan
            in_std_burstbytes = np.nan
            in_max_burstbytes = np.nan
            in_min_burstbytes = np.nan
                
        return [
            out_mean_burstbytes,
            in_mean_burstbytes,
            out_median_burstbytes,
            in_median_burstbytes,
            out_25per_burstbytes,
            in_25per_burstbytes,
            out_75per_burstbytes,
            in_75per_burstbytes,
            out_90per_burstbytes,
            in_90per_burstbytes,
            out_std_burstbytes,
            in_std_burstbytes,
            out_max_burstbytes,
            in_max_burstbytes,
            out_min_burstbytes,
            in_min_burstbytes
        ]
    def __extractBurstTime(self, allBursts, outgoingBursts, incomingBursts):
        '''
        FEATURES
        out_mean_bursttime
        in_mean_bursttime
        out_median_bursttime
        in_median_bursttime
        out_25per_bursttime
        in_25per_bursttime
        out_75per_bursttime
        in_75per_bursttime
        out_90per_bursttime
        in_90per_bursttime
        out_std_bursttime
        in_std_bursttime
        out_max_bursttime
        in_max_bursttime
        out_min_bursttime
        in_min_bursttime
        '''
        try:
            out_bursttime = outgoingBursts['End Time'].values - outgoingBursts['Start Time'].values
            out_bursttime = self.__convertTimeStamp(out_bursttime)
        except Exception as e:
            print(e)
            out_bursttime = []
        try:
            in_bursttime = incomingBursts['End Time'].values - incomingBursts['Start Time'].values
            in_bursttime = self.__convertTimeStamp(in_bursttime)
        except Exception as e:
            print(e)
            in_bursttime = []
        
        if len(out_bursttime):
            out_mean_bursttime = np.mean(out_bursttime)
            out_median_bursttime = np.median(out_bursttime)
            out_25per_bursttime = np.percentile(out_bursttime, 25)
            out_75per_bursttime = np.percentile(out_bursttime, 75)
            out_90per_bursttime = np.percentile(out_bursttime, 90)
            out_std_bursttime = np.std(out_bursttime)
            out_max_bursttime = np.max(out_bursttime)
            out_min_bursttime = np.min(out_bursttime)
        else:
            out_mean_bursttime = np.nan
            out_median_bursttime = np.nan
            out_25per_bursttime = np.nan
            out_75per_bursttime = np.nan
            out_90per_bursttime = np.nan
            out_std_bursttime = np.nan
            out_max_bursttime = np.nan
            out_min_bursttime = np.nan
        
        if len(in_bursttime):
            in_mean_bursttime = np.mean(in_bursttime)
            in_median_bursttime = np.median(in_bursttime)
            in_25per_bursttime = np.percentile(in_bursttime, 25)
            in_75per_bursttime = np.percentile(in_bursttime, 75)
            in_90per_bursttime = np.percentile(in_bursttime, 90)
            in_std_bursttime = np.std(in_bursttime)
            in_max_bursttime = np.max(in_bursttime)
            in_min_bursttime = np.min(in_bursttime)
        else:
            in_mean_bursttime = np.nan
            in_median_bursttime = np.nan
            in_25per_bursttime = np.nan
            in_75per_bursttime = np.nan
            in_90per_bursttime = np.nan
            in_std_bursttime = np.nan
            in_max_bursttime = np.nan
            in_min_bursttime = np.nan
                
        return [
            out_mean_bursttime,
            in_mean_bursttime,
            out_median_bursttime,
            in_median_bursttime,
            out_25per_bursttime,
            in_25per_bursttime,
            out_75per_bursttime,
            in_75per_bursttime,
            out_90per_bursttime,
            in_90per_bursttime,
            out_std_bursttime,
            in_std_bursttime,
            out_max_bursttime,
            in_max_bursttime,
            out_min_bursttime,
            in_min_bursttime
        ]

# Aggregation Functions    
    def __extractDictFeatures(self, allPackets, outgoingPackets, incomingPackets):
        returnArray = []

        returnArray.extend(self.extractDevicePortCount(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractExternalPortCount(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractContactedHostName(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractContactedIP(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractProtocols(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractPacketSizes(allPackets, outgoingPackets, incomingPackets))
        returnArray.extend(self.extractPingPongPairs(allPackets, outgoingPackets, incomingPackets))
        
        return returnArray
    def extractAllFeatures(self, window):
    # Init
        featureArray = []
        incomingPackets = window[window['PacketType'] == 'Incoming']
        outgoingPackets = window[window['PacketType'] == 'Outgoing']
        
        bursts = self.groupBurstPackets(window)
        outgoingBursts = bursts[bursts['Type'] == 'Outgoing']
        incomingBursts = bursts[bursts['Type'] == 'Incoming']
    
    # Extract Simple Features 
        featureArray.extend(self.__extractTotalPkts(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractTotalBytes(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractUniqueLen(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractPacketPercentage(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractTCPFlags(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractProtocols(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractUniqueProtocols(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractHostNameIP(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractUniqueSrcDstPorts(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extract80and443Features(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractInterPacketDelay(window, outgoingPackets, incomingPackets))
        featureArray.extend(self.__extractProtocolBasedInterPacketDelay(window, outgoingPackets, incomingPackets))
    
    # Extract Burst Features 
        featureArray.extend(self.__extractInterBurstDelay(bursts, outgoingBursts, incomingBursts))
        featureArray.extend(self.__extractBurstNumPackets(bursts, outgoingBursts, incomingBursts))
        featureArray.extend(self.__extractBurstBytes(bursts, outgoingBursts, incomingBursts))
        featureArray.extend(self.__extractBurstTime(bursts, outgoingBursts, incomingBursts))

    # Dict Features
        featureArray.extend(self.__extractDictFeatures(window, outgoingPackets, incomingPackets))
    
    # Return Values
        return pd.Series(featureArray)

    def makeSimpleWindows(self, packets):
        if packets.shape[0]:
            return packets.groupby([pd.Grouper(key='Time', freq=self.__winSize, axis=1),'Device'])
        else:
            return None

    def run(self, packets, reset_index=True):
    # Init 
        self.__featureData = pd.DataFrame([], columns=self.__featureNames)

    # Make Windows
        if self.__winType.lower() == 'simple':
            windows = self.makeSimpleWindows(packets)
        else:
            raise Exception('Feature Not Yet Implemented')
        if type(None) != type(windows):
            print('run_FeatureExtraction: Extracting Features', flush=True)
            self.__featureData = windows.apply(self.extractAllFeatures)
            

    # Extract Features Per Window
    
    # Wrap Up
        self.__featureData.columns = self.__featureNames
        if reset_index:
            self.__featureData = self.__featureData.reset_index()
        try:
            self.__featureData = self.__featureData.drop(columns=['Time'])
        except KeyError as e:
            print('run_FeatureExtraction', e)
    
    # Return Values
        return self.__featureData