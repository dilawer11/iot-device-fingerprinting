features = {
    'out_totalpkts': 'Total Outgoing Packets',
    'in_totalpkts': 'Total Incoming Packets',
    'out_totalbytes': 'Total Outgoing Bytes',
    'in_totalbytes': 'Total Incoming Bytes',
    'mean_out_uniquelen': 'Mean of unique length outgoing packets',
    'mean_in_uniquelen': 'Mean of unique length incoming packets',
    'median_out_uniquelen': 'Median of unique length outgoing packets',
    'median_in_uniquelen': 'Median of unique length incoming packets',
    '25per_out_uniquelen': '25th percentile of unique length outgoing packets',
    '25per_in_uniquelen': '25th percentile of unique length incoming packets',
    '75per_out_uniquelen': '75th percentile of unique length outgoing packets',
    '75per_in_uniquelen': '75th percentile of unique length incoming packets',
    '90per_out_uniquelen': '90th percentile of unique length outgoing packets',
    '90per_in_uniquelen': '90th percentile of unique length incoming packets',
    'len_out_uniquelen': 'Total number of unique length outgoing packets',
    'len_in_uniquelen': 'Total number of unique length incoming packets',
    'max_out_len': 'Maximum length outgoing packet',
    'max_in_len': 'Maximum length incoming packet',
    'min_out_len': 'Minimum length outgoing packet',
    'min_in_len': 'Minimum length incoming packet',
    'out_percentage': 'Percentage of outgoing packets',
    'in_percentage': 'Percentage of incoming packets',
    'out_tcpack_percentage': 'Percentage of Outgoing TCP Packets with ACK flag set',
    'out_tcpsyn_percentage': 'Percentage of Outgoing TCP Packets with SYN flag set',
    'out_tcpfin_percentage': 'Percentage of Outgoing TCP Packets with FIN flag set',
    'out_tcprst_percentage': 'Percentage of Outgoing TCP Packets with RST flag set',
    'out_tcppsh_percentage': 'Percentage of Outgoing TCP Packets with PUSH flag set',
    'out_tcpurg_percentage': 'Percentage of Outgoing TCP Packets with URG flag set',
    'in_tcpack_percentage': 'Percentage of Incoming TCP Packets with ACK flag set',
    'in_tcpsyn_percentage': 'Percentage of Incoming TCP Packets with SYN flag set',
    'in_tcpfin_percentage': 'Percentage of Incoming TCP Packets with FIN flag set',
    'in_tcprst_percentage': 'Percentage of Incoming TCP Packets with RST flag set',
    'in_tcppsh_percentage': 'Percentage of Incoming TCP Packets with PUSH flag set',
    'in_tcpurg_percentage': 'Percentage of Incoming TCP Packets with URG flag set',
    'out_tls1pkts_percentage': 'Percentage of outgoing packets using TLSv1 protocol',
    'in_tls1pkts_percentage': 'Percentage of incoming packets using TLSv1 protocol',
    'out_tls12pkts_percentage': 'Percentage of outgoing packets using TLSv1.2 protocol',
    'in_tls12pkts_percentage': 'Percentage of incoming packets using TLSv1.2 protocol',
    'out_tcppkts_percentage': 'Percentage of outgoing packets using TCP protocol',
    'in_tcppkts_percentage': 'Percentage of incoming packets using TCP protocol',
    'out_udppkts_percentage': 'Percentage of outgoing packets using UDP protocol',
    'in_udppkts_percentage': 'Percentage of incoming packets using UDP protocol',
    'out_dnspkts_percentage': 'Percentage of outgoing packets using DNS protocol',
    'in_dnspkts_percentage': 'Percentage of incoming packets using DNS protocol',
    'out_ssdppkts_percentage': 'Percentage of outgoing packets using SSDP protocol',
    'in_ssdppkts_percentage': 'Percentage of incoming packets using SSDP protocol',
    'out_sslpkts_percentage': 'Percentage of outgoing packets using SSL protocol',
    'in_sslpkts_percentage': 'Percentage of incoming packets using SSL protocol',
    'out_icmppkts_percentage': 'Percentage of outgoing packets using ICMP protocol',
    'in_icmppkts_percentage': 'Percentage of incoming packets using ICMP protocol',
    'out_ntppkts_percentage': 'Percentage of outgoing packets using NTP protocol',
    'in_ntppkts_percentage': 'Percentage of incoming packets using NTP protocol',
    'out_numuniqueprotocol': 'Unique protocols used by outgoing packets',
    'in_numuniqueprotocol': 'Unique protocols used by incoming packets',
    'num_unique_ip': 'Unique IPs contacted',
    'num_unique_ip_3octet' : 'Unique IPs contacted but upto 3 octets (e.g 192.128.1.1 and 192.128.1.2 are considered same)',
    'num_unique_hostname' : 'Unique hostnames contacted TLD+1 for upto 3 level of domain and TLD+2 for 4(TLD+3) or more ',
    'out_numuniquesrcport':'Total number of unique source ports used by outgoing packets', 
    'in_numuniquesrcport': 'Total number of unique source ports used by incoming packets',
    'out_numuniquedstport':'Total number of unique destination ports used by outgoing packets',
    'in_numuniquedstport': 'Total number of unique destination ports used by incoming packets',
    
    'pkts_80_443_percentage' : 'Percentage Packets to and from Port 80 and 443',
    'byte_per_pkt_80_443': 'Bytes Per Packet to and from Port 80 and 443',

    'mean_interpktdelay': 'Mean delay between packets',
    'median_interpktdelay': 'Median delay between packets',
    '25per_interpktdelay': '25th percentile of delay between packets',
    '75per_interpktdelay': '75th percentile of delay between packets',
    '90per_interpktdelay': '90th percentile of delay between packets',
    'std_interpktdelay': 'Standard Deviation of delays between packets',
    'max_interpktdelay': 'Maximum delay between packets',
    'min_interpktdelay': 'Minimum delay between packets',

    'out_mean_inter_tls_pkt_delay': 'Mean of Inter TLS Packet Delay in Outgoing Traffic',          
    'out_median_inter_tls_pkt_delay': 'Median of Inter TLS Packet Delay in Outgoing Traffic',       
    'out_25per_inter_tls_pkt_delay': '25th Percentile of Inter TLS Packet Delay in Outgoing Traffic', 
    'out_75per_inter_tls_pkt_delay': '75th Percentile of Inter TLS Packet Delay in Outgoing Traffic', 
    'out_90per_inter_tls_pkt_delay': '90th Percentile of Inter TLS Packet Delay in Outgoing Traffic', 
    'out_std_inter_tls_pkt_delay': 'Standard Deviation of Inter TLS Packet Delay in Outgoing Traffic', 
    'out_max_inter_tls_pkt_delay': 'Maximum of Inter TLS Packet Delay in Outgoing Traffic', 
    'out_min_inter_tls_pkt_delay': 'Minimum of Inter TLS Packet Delay in Outgoing Traffic',
    'in_mean_inter_tls_pkt_delay': 'Mean of Inter TLS Packet Delay in Incoming Traffic',             
    'in_median_inter_tls_pkt_delay': 'Median of Inter TLS Packet Delay in Incoming Traffic',           
    'in_25per_inter_tls_pkt_delay': '25th Percentile of Inter TLS Packet Delay in Incoming Traffic',
    'in_75per_inter_tls_pkt_delay': '75th Percentile of Inter TLS Packet Delay in Incoming Traffic',
    'in_90per_inter_tls_pkt_delay': '90th Percentile of Inter TLS Packet Delay in Incoming Traffic',
    'in_std_inter_tls_pkt_delay': 'Standard Deviation of Inter TLS Packet Delay in Incoming Traffic',
    'in_max_inter_tls_pkt_delay': 'Maximum of Inter TLS Packet Delay in Incoming Traffic',
    'in_min_inter_tls_pkt_delay': 'Minimum of Inter TLS Packet Delay in Incoming Traffic',

    'out_mean_inter_tcp_pkt_delay': 'Mean of Inter TCP Packet Delay in Outgoing Traffic',          
    'out_median_inter_tcp_pkt_delay': 'Median of Inter TCP Packet Delay in Outgoing Traffic',       
    'out_25per_inter_tcp_pkt_delay': '25th Percentile of Inter TCP Packet Delay in Outgoing Traffic', 
    'out_75per_inter_tcp_pkt_delay': '75th Percentile of Inter TCP Packet Delay in Outgoing Traffic', 
    'out_90per_inter_tcp_pkt_delay': '90th Percentile of Inter TCP Packet Delay in Outgoing Traffic', 
    'out_std_inter_tcp_pkt_delay': 'Standard Deviation of Inter TCP Packet Delay in Outgoing Traffic', 
    'out_max_inter_tcp_pkt_delay': 'Maximum of Inter TCP Packet Delay in Outgoing Traffic', 
    'out_min_inter_tcp_pkt_delay': 'Minimum of Inter TCP Packet Delay in Outgoing Traffic',
    'in_mean_inter_tcp_pkt_delay': 'Mean of Inter TCP Packet Delay in Incoming Traffic',             
    'in_median_inter_tcp_pkt_delay': 'Median of Inter TCP Packet Delay in Incoming Traffic',           
    'in_25per_inter_tcp_pkt_delay': '25th Percentile of Inter TCP Packet Delay in Incoming Traffic',
    'in_75per_inter_tcp_pkt_delay': '75th Percentile of Inter TCP Packet Delay in Incoming Traffic',
    'in_90per_inter_tcp_pkt_delay': '90th Percentile of Inter TCP Packet Delay in Incoming Traffic',
    'in_std_inter_tcp_pkt_delay': 'Standard Deviation of Inter TCP Packet Delay in Incoming Traffic',
    'in_max_inter_tcp_pkt_delay': 'Maximum of Inter TCP Packet Delay in Incoming Traffic',
    'in_min_inter_tcp_pkt_delay': 'Minimum of Inter TCP Packet Delay in Incoming Traffic',

    'out_mean_inter_dns_pkt_delay': 'Mean of Inter DNS Packet Delay in Outgoing Traffic',          
    'out_median_inter_dns_pkt_delay': 'Median of Inter DNS Packet Delay in Outgoing Traffic',       
    'out_25per_inter_dns_pkt_delay': '25th Percentile of Inter DNS Packet Delay in Outgoing Traffic', 
    'out_75per_inter_dns_pkt_delay': '75th Percentile of Inter DNS Packet Delay in Outgoing Traffic', 
    'out_90per_inter_dns_pkt_delay': '90th Percentile of Inter DNS Packet Delay in Outgoing Traffic', 
    'out_std_inter_dns_pkt_delay': 'Standard Deviation of Inter DNS Packet Delay in Outgoing Traffic', 
    'out_max_inter_dns_pkt_delay': 'Maximum of Inter DNS Packet Delay in Outgoing Traffic', 
    'out_min_inter_dns_pkt_delay': 'Minimum of Inter DNS Packet Delay in Outgoing Traffic',
    'in_mean_inter_dns_pkt_delay': 'Mean of Inter DNS Packet Delay in Incoming Traffic',          
    'in_median_inter_dns_pkt_delay': 'Median of Inter DNS Packet Delay in Incoming Traffic',       
    'in_25per_inter_dns_pkt_delay': '25th Percentile of Inter DNS Packet Delay in Incoming Traffic', 
    'in_75per_inter_dns_pkt_delay': '75th Percentile of Inter DNS Packet Delay in Incoming Traffic', 
    'in_90per_inter_dns_pkt_delay': '90th Percentile of Inter DNS Packet Delay in Incoming Traffic', 
    'in_std_inter_dns_pkt_delay': 'Standard Deviation of Inter DNS Packet Delay in Incoming Traffic', 
    'in_max_inter_dns_pkt_delay': 'Maximum of Inter DNS Packet Delay in Incoming Traffic', 
    'in_min_inter_dns_pkt_delay': 'Minimum of Inter DNS Packet Delay in Incoming Traffic',

    'out_mean_inter_udp_pkt_delay': 'Mean of Inter UDP Packet Delay in Outgoing Traffic',          
    'out_median_inter_udp_pkt_delay': 'Median of Inter UDP Packet Delay in Outgoing Traffic',       
    'out_25per_inter_udp_pkt_delay': '25th Percentile of Inter UDP Packet Delay in Outgoing Traffic', 
    'out_75per_inter_udp_pkt_delay': '75th Percentile of Inter UDP Packet Delay in Outgoing Traffic', 
    'out_90per_inter_udp_pkt_delay': '90th Percentile of Inter UDP Packet Delay in Outgoing Traffic', 
    'out_std_inter_udp_pkt_delay': 'Standard Deviation of Inter UDP Packet Delay in Outgoing Traffic', 
    'out_max_inter_udp_pkt_delay': 'Maximum of Inter UDP Packet Delay in Outgoing Traffic', 
    'out_min_inter_udp_pkt_delay': 'Minimum of Inter UDP Packet Delay in Outgoing Traffic',
    'in_mean_inter_udp_pkt_delay': 'Mean of Inter UDP Packet Delay in Incoming Traffic',             
    'in_median_inter_udp_pkt_delay': 'Median of Inter UDP Packet Delay in Incoming Traffic',           
    'in_25per_inter_udp_pkt_delay': '25th Percentile of Inter UDP Packet Delay in Incoming Traffic',
    'in_75per_inter_udp_pkt_delay': '75th Percentile of Inter UDP Packet Delay in Incoming Traffic',
    'in_90per_inter_udp_pkt_delay': '90th Percentile of Inter UDP Packet Delay in Incoming Traffic',
    'in_std_inter_udp_pkt_delay': 'Standard Deviation of Inter UDP Packet Delay in Incoming Traffic',
    'in_max_inter_udp_pkt_delay': 'Maximum of Inter UDP Packet Delay in Incoming Traffic',
    'in_min_inter_udp_pkt_delay': 'Minimum of Inter UDP Packet Delay in Incoming Traffic',

    'out_mean_inter_ntp_pkt_delay': 'Mean of Inter NTP Packet Delay in Outgoing Traffic',          
    'out_median_inter_ntp_pkt_delay': 'Median of Inter NTP Packet Delay in Outgoing Traffic',       
    'out_25per_inter_ntp_pkt_delay': '25th Percentile of Inter NTP Packet Delay in Outgoing Traffic', 
    'out_75per_inter_ntp_pkt_delay': '75th Percentile of Inter NTP Packet Delay in Outgoing Traffic', 
    'out_90per_inter_ntp_pkt_delay': '90th Percentile of Inter NTP Packet Delay in Outgoing Traffic', 
    'out_std_inter_ntp_pkt_delay': 'Standard Deviation of Inter NTP Packet Delay in Outgoing Traffic', 
    'out_max_inter_ntp_pkt_delay': 'Maximum of Inter NTP Packet Delay in Outgoing Traffic', 
    'out_min_inter_ntp_pkt_delay': 'Minimum of Inter NTP Packet Delay in Outgoing Traffic',
    'in_mean_inter_ntp_pkt_delay': 'Mean of Inter NTP Packet Delay in Incoming Traffic',          
    'in_median_inter_ntp_pkt_delay': 'Median of Inter NTP Packet Delay in Incoming Traffic',       
    'in_25per_inter_ntp_pkt_delay': '25th Percentile of Inter NTP Packet Delay in Incoming Traffic', 
    'in_75per_inter_ntp_pkt_delay': '75th Percentile of Inter NTP Packet Delay in Incoming Traffic', 
    'in_90per_inter_ntp_pkt_delay': '90th Percentile of Inter NTP Packet Delay in Incoming Traffic', 
    'in_std_inter_ntp_pkt_delay': 'Standard Deviation of Inter NTP Packet Delay in Incoming Traffic', 
    'in_max_inter_ntp_pkt_delay': 'Maximum of Inter NTP Packet Delay in Incoming Traffic', 
    'in_min_inter_ntp_pkt_delay': 'Minimum of Inter NTP Packet Delay in Incoming Traffic',
 
    'out_mean_interburstdelay':'Mean delay between outgoing bursts',
    'in_mean_interburstdelay': 'Mean delay between incoming bursts',
    'out_median_interburstdelay':'Median delay between outgoing bursts',
    'in_median_interburstdelay': 'Median delay between incoming bursts',
    'out_25per_interburstdelay':'25th percentile delay between outgoing bursts',
    'in_25per_interburstdelay': '25th percentile delay between incoming bursts',
    'out_75per_interburstdelay':'75th percentile delay between outgoing bursts',
    'in_75per_interburstdelay': '75th percentile delay between incoming bursts',
    'out_90per_interburstdelay':'90th percentile delay between outgoing bursts',
    'in_90per_interburstdelay': '90th percentile delay between incoming bursts',
    'out_std_interburstdelay':'Standard Deviation of delay between outgoing bursts',
    'in_std_interburstdelay': 'Standard Deviation of delay between incoming bursts',
    'out_max_interburstdelay':'Maximum delay between outgoing bursts',
    'in_max_interburstdelay': 'Maximum delay between incoming bursts',
    'out_min_interburstdelay':'Minimum delay between outgoing bursts',
    'in_min_interburstdelay': 'Minimum delay between incoming bursts',
    'out_mean_burstnumpkts':'Mean number of packets in outgoing bursts',
    'in_mean_burstnumpkts': 'Mean number of packets in incoming bursts',
    'out_median_burstnumpkts':'Median number of packets in outgoing bursts',
    'in_median_burstnumpkts': 'Median number of packets in incoming bursts',
    'out_25per_burstnumpkts':'25th percentile number of packets in outgoing bursts',
    'in_25per_burstnumpkts': '25th percentile number of packets in incoming bursts',
    'out_75per_burstnumpkts':'75th percentile number of packets in outgoing bursts',
    'in_75per_burstnumpkts': '75th percentile number of packets in incoming bursts',
    'out_90per_burstnumpkts':'90th percentile number of packets in outgoing bursts',
    'in_90per_burstnumpkts': '90th percentile number of packets in incoming bursts',
    'out_std_burstnumpkts':'Standard Deviation of number of packets in outgoing bursts',
    'in_std_burstnumpkts': 'Standard Deviation of number of packets in incoming bursts',
    'out_max_burstnumpkts':'Maximum number of packets in outgoing bursts',
    'in_max_burstnumpkts': 'Maximum number of packets in incoming bursts',
    'out_min_burstnumpkts':'Minimum number of packets in outgoing bursts',
    'in_min_burstnumpkts': 'Minimum number of packets in incoming bursts',
    'out_mean_burstbytes':'Mean number of bytes in outgoing bursts',
    'in_mean_burstbytes': 'Mean number of bytes in incoming bursts',
    'out_median_burstbytes':'Median number of bytes in outgoing bursts',
    'in_median_burstbytes': 'Median number of bytes in incoming bursts',
    'out_25per_burstbytes':'25th percentile number of bytes in outgoing bursts',
    'in_25per_burstbytes': '25th percentile number of bytes in incoming bursts',
    'out_75per_burstbytes':'75th percentile number of bytes in outgoing bursts',
    'in_75per_burstbytes': '75th percentile number of bytes in incoming bursts',
    'out_90per_burstbytes':'90th percentile number of bytes in outgoing bursts',
    'in_90per_burstbytes': '90th percentile number of bytes in incoming bursts',
    'out_std_burstbytes':'Standard Deviation of number of bytes in outgoing bursts',
    'in_std_burstbytes': 'Standard Deviation of number of bytes in incoming bursts',
    'out_max_burstbytes':'Maximum number of bytes in outgoing bursts',
    'in_max_burstbytes': 'Maximum number of bytes in incoming bursts',
    'out_min_burstbytes':'Minimum number of bytes in outgoing bursts',
    'in_min_burstbytes': 'Minimum number of bytes in incoming bursts',
    'out_mean_bursttime':'Mean ongoing time of outgoing bursts',
    'in_mean_bursttime': 'Mean ongoing time of incoming bursts',
    'out_median_bursttime':'Median ongoing time of outgoing bursts',
    'in_median_bursttime': 'Median ongoing time of incoming bursts',
    'out_25per_bursttime':'25th percentile ongoing time of outgoing bursts',
    'in_25per_bursttime': '25th percentile ongoing time of incoming bursts',
    'out_75per_bursttime':'75th percentile ongoing time of outgoing bursts',
    'in_75per_bursttime': '75th percentile ongoing time of incoming bursts',
    'out_90per_bursttime':'90th percentile ongoing time of outgoing bursts',
    'in_90per_bursttime': '90th percentile ongoing time of incoming bursts',
    'out_std_bursttime':'Standard Deviation of ongoing time of outgoing bursts',
    'in_std_bursttime': 'Standard Deviation of ongoing time of incoming bursts',
    'out_max_bursttime':'Maximum ongoing time of outgoing bursts',
    'in_max_bursttime': 'Maximum ongoing time of incoming bursts',
    'out_min_bursttime':'Minimum ongoing time of outgoing bursts',
    'in_min_bursttime': 'Minimum ongoing time of incoming bursts',
    
    # Dict Features
    'Device Port Dict': 'Series with Value Counts of Device Ports used',
    'Device Port TLSTCP': 'Value counts of Device Ports used in TLS and TCP Packets',
    'Device Port DNS': 'Value counts of Device Ports used in DNS Packets',
    'Device Port UDP': 'Value counts of Device Ports used in UDP Packets',
    'Device Port NTP': 'Value counts of Device Ports used in NTP Packets',
    'External Port Dict': 'Series with Value Counts of External IP Ports used',
    'External Port TLSTCP': 'Value counts of External Ports used in TLS and TCP Packets',
    'External Port DNS': 'Value counts of External Ports used in DNS Packets',
    'External Port UDP': 'Value counts of External Ports used in UDP Packets',
    'External Port NTP': 'Value counts of External Ports used in NTP Packets',
    'Hostname Dict': 'Series with Value Counts of Reverse DNS Lookup Hostnames',
    'Hostname TLSTCP': 'Value counts of Hostnames contacted in TLS and TCP Traffic',
    'Hostname DNS': 'Value counts of Hostnames contacted in DNS Traffic',
    'Hostname UDP': 'Value counts of Hostnames contacted in UDP Traffic',
    'Hostname NTP': 'Value counts of Hostnames contacted in NTP Traffic',
    'IP Dict': 'Series with Value Counts of IP (3 octet) contacted',
    'IP TLSTCP': 'Value counts of IP (3octet) contacted in TLS and TCP Traffic',
    'IP DNS': 'Value counts of IP (3octet) contacted in DNS Traffic',
    'IP UDP': 'Value counts of IP (3octet) contacted in UDP Traffic',
    'IP NTP': 'Value counts of IP (3octet) contacted in NTP Traffic',
    'Protocol Dict': 'Series with Value Counts of Protocols',
    'Packet Sizes TLSTCP': 'Series with Value Counts of TLS and TCP Packets',
    'Packet Sizes DNS': 'Series with Value Counts of DNS Packets',
    'Packet Sizes UDP': 'Series with Value Counts of UDP Packets',
    'Packet Sizes NTP': 'Series with Value Counts of NTP Packets',
    'Ping Pong Dict': 'Series with Value Counts of Ping Pong All Protocols',
    'TLSTCP Ping Pong Sizes': 'Series with Value Counts of Ping Pong Sizes TLS and TCP',
    'DNS Ping Pong Sizes': 'Series with Value Counts of Ping Pong Sizes DNS',
    'UDP Ping Pong Sizes': 'Series with Value Counts of Ping Pong Sizes UDP',
    'NTP Ping Pong Sizes': 'Series with Value Counts of Ping Pong Sizes NTP',
}
featureGroups = {
    'burstTime' : [ 
        'out_mean_bursttime',
        'in_mean_bursttime',
        'out_median_bursttime',
        'in_median_bursttime',
        'out_25per_bursttime',
        'in_25per_bursttime',
        'out_75per_bursttime',
        'in_75per_bursttime',
        'out_90per_bursttime',
        'in_90per_bursttime',
        'out_std_bursttime',
        'in_std_bursttime',
        'out_max_bursttime',
        'in_max_bursttime',
        'out_min_bursttime',
        'in_min_bursttime'],
    'burstTime-min' : [ 
        'out_mean_bursttime',
        'in_mean_bursttime',
        'out_median_bursttime',
        'in_median_bursttime',
        'out_std_bursttime',
        'in_std_bursttime',
        'out_max_bursttime',
        'in_max_bursttime',
        'out_min_bursttime',
        'in_min_bursttime'],
    'burstBytes' : [ 
        'out_mean_burstbytes',
        'in_mean_burstbytes',
        'out_median_burstbytes',
        'in_median_burstbytes',
        'out_25per_burstbytes',
        'in_25per_burstbytes',
        'out_75per_burstbytes',
        'in_75per_burstbytes',
        'out_90per_burstbytes',
        'in_90per_burstbytes',
        'out_std_burstbytes',
        'in_std_burstbytes',
        'out_max_burstbytes',
        'in_max_burstbytes',
        'out_min_burstbytes',
        'in_min_burstbytes'],
    'burstBytes-min' : [ 
        'out_mean_burstbytes',
        'in_mean_burstbytes',
        'out_median_burstbytes',
        'in_median_burstbytes',
        'out_std_burstbytes',
        'in_std_burstbytes',
        'out_max_burstbytes',
        'in_max_burstbytes',
        'out_min_burstbytes',
        'in_min_burstbytes'],
    'burstPackets': [
        'out_mean_burstnumpkts',
        'in_mean_burstnumpkts',
        'out_median_burstnumpkts',
        'in_median_burstnumpkts',
        'out_25per_burstnumpkts',
        'in_25per_burstnumpkts',
        'out_75per_burstnumpkts',
        'in_75per_burstnumpkts',
        'out_90per_burstnumpkts',
        'in_90per_burstnumpkts',
        'out_std_burstnumpkts',
        'in_std_burstnumpkts',
        'out_max_burstnumpkts',
        'in_max_burstnumpkts',
        'out_min_burstnumpkts',
        'in_min_burstnumpkts'],
    'burstPackets-min': [
        'out_mean_burstnumpkts',
        'in_mean_burstnumpkts',
        'out_median_burstnumpkts',
        'in_median_burstnumpkts',
        'out_std_burstnumpkts',
        'in_std_burstnumpkts',
        'out_max_burstnumpkts',
        'in_max_burstnumpkts',
        'out_min_burstnumpkts',
        'in_min_burstnumpkts'],
    'burstDelay':[
        'out_mean_interburstdelay',
        'in_mean_interburstdelay',
        'out_median_interburstdelay',
        'in_median_interburstdelay',
        'out_25per_interburstdelay',
        'in_25per_interburstdelay',
        'out_75per_interburstdelay',
        'in_75per_interburstdelay',
        'out_90per_interburstdelay',
        'in_90per_interburstdelay',
        'out_std_interburstdelay',
        'in_std_interburstdelay',
        'out_max_interburstdelay',
        'in_max_interburstdelay',
        'out_min_interburstdelay',
        'in_min_interburstdelay'],
    'burstDelay-min':[
        'out_mean_interburstdelay',
        'in_mean_interburstdelay',
        'out_median_interburstdelay',
        'in_median_interburstdelay',
        'out_std_interburstdelay',
        'in_std_interburstdelay',
        'out_max_interburstdelay',
        'in_max_interburstdelay',
        'out_min_interburstdelay',
        'in_min_interburstdelay'],
    'packetDelay':[
        'mean_interpktdelay',
        'median_interpktdelay',
        '25per_interpktdelay',
        '75per_interpktdelay',
        '90per_interpktdelay',
        'std_interpktdelay',
        'max_interpktdelay',
        'min_interpktdelay'],
    'packetDelay-min':[
        'mean_interpktdelay',
        'median_interpktdelay',
        'std_interpktdelay',
        'max_interpktdelay',
        'min_interpktdelay'],
    'uniqueIPbased':[
        'num_unique_ip',
        'num_unique_ip_3octet',
        'num_unique_hostname'],
    'uniquePorts':[
        'in_numuniquesrcport',
        'out_numuniquedstport'],
    'protocols':[
        'out_tls1pkts_percentage',
        'in_tls1pkts_percentage',
        'out_tls12pkts_percentage',
        'in_tls12pkts_percentage',
        'out_tcppkts_percentage',
        'in_tcppkts_percentage',
        'out_udppkts_percentage',
        'in_udppkts_percentage',
        'out_dnspkts_percentage',
        'in_dnspkts_percentage',
        'out_ssdppkts_percentage',
        'in_ssdppkts_percentage',
        'out_sslpkts_percentage',
        'in_sslpkts_percentage',
        'out_icmppkts_percentage',
        'in_icmppkts_percentage',
        'out_ntppkts_percentage',
        'in_ntppkts_percentage',
        'out_numuniqueprotocol',
        'in_numuniqueprotocol'],
    'protocols-min':[
        'out_tls1pkts_percentage',
        'in_tls1pkts_percentage',
        'out_tls12pkts_percentage',
        'in_tls12pkts_percentage',
        'out_tcppkts_percentage',
        'in_tcppkts_percentage',
        'out_udppkts_percentage',
        'in_udppkts_percentage',
        'out_dnspkts_percentage',
        'in_dnspkts_percentage',
        'out_numuniqueprotocol',
        'in_numuniqueprotocol'],
    'totalPackets':[
        'out_totalpkts',
        'in_totalpkts'],
    'totalBytes':[
        'out_totalbytes',
        'in_totalbytes'],
    'uniquePacketLengths':[
        'mean_out_uniquelen',
        'mean_in_uniquelen',
        'median_out_uniquelen',
        'median_in_uniquelen',
        '25per_out_uniquelen',
        '25per_in_uniquelen',
        '75per_out_uniquelen',
        '75per_in_uniquelen',
        '90per_out_uniquelen',
        '90per_in_uniquelen',
        'len_out_uniquelen',
        'len_in_uniquelen',
        'max_out_len',
        'max_in_len',
        'min_out_len',
        'min_in_len'],
    'uniquePacketLengths-min':[
        'mean_out_uniquelen',
        'mean_in_uniquelen',
        'median_out_uniquelen',
        'median_in_uniquelen',
        'len_out_uniquelen',
        'len_in_uniquelen',
        'max_out_len',
        'max_in_len',
        'min_out_len',
        'min_in_len'],
    'packetInOutRatio':[
        'out_percentage',
        'in_percentage'],
    'tcpFlags':[
        'out_tcpack_percentage',
        'out_tcpsyn_percentage',
        'out_tcpfin_percentage',
        'out_tcprst_percentage',
        'out_tcppsh_percentage',
        'out_tcpurg_percentage',
        'in_tcpack_percentage',
        'in_tcpsyn_percentage',
        'in_tcpfin_percentage',
        'in_tcprst_percentage',
        'in_tcppsh_percentage',
        'in_tcpurg_percentage'],
    '80-443':[
        'pkts_80_443_percentage',
        'byte_per_pkt_80_443'
    ]
}
dictFeatures = {
    'Device Port Dict': 'Value Counts of Device Ports used',
    'Device Port TLSTCP': 'Value counts of Device Ports used in TLS and TCP Packets',
    'Device Port DNS': 'Value counts of Device Ports used in DNS Packets',
    'Device Port UDP': 'Value counts of Device Ports used in UDP Packets',
    'Device Port NTP': 'Value counts of Device Ports used in NTP Packets',
    'External Port Dict': 'Value Counts of External IP Ports used',
    'External Port TLSTCP': 'Value counts of External Ports used in TLS and TCP Packets',
    'External Port DNS': 'Value counts of External Ports used in DNS Packets',
    'External Port UDP': 'Value counts of External Ports used in UDP Packets',
    'External Port NTP': 'Value counts of External Ports used in NTP Packets',
    'Hostname Dict': 'Value Counts of Reverse DNS Lookup Hostnames',
    'Hostname TLSTCP': 'Value counts of Hostnames contacted in TLS and TCP Traffic',
    'Hostname DNS': 'Value counts of Hostnames contacted in DNS Traffic',
    'Hostname UDP': 'Value counts of Hostnames contacted in UDP Traffic',
    'Hostname NTP': 'Value counts of Hostnames contacted in NTP Traffic',
    'IP Dict': 'Value Counts of IP (3 octet) contacted',
    'IP TLSTCP': 'Value counts of IP (3octet) contacted in TLS and TCP Traffic',
    'IP DNS': 'Value counts of IP (3octet) contacted in DNS Traffic',
    'IP UDP': 'Value counts of IP (3octet) contacted in UDP Traffic',
    'IP NTP': 'Value counts of IP (3octet) contacted in NTP Traffic',
    'Protocol Dict': 'Value Counts of Protocols',
    'Packet Sizes TLSTCP': 'Value Counts of TLS and TCP Packet Sizes',
    'Packet Sizes DNS': 'Value Counts of DNS Packet Sizes',
    'Packet Sizes UDP': 'Value Counts of UDP Packet Sizes',
    'Packet Sizes NTP': 'Value Counts of NTP Packet Sizes',
    'Ping Pong Dict': 'Value Counts of Ping Pong All Protocols',
    'TLSTCP Ping Pong Sizes': 'Value Counts of Ping Pong Sizes TLS and TCP',
    'DNS Ping Pong Sizes': 'Value Counts of Ping Pong Sizes DNS',
    'UDP Ping Pong Sizes': 'Value Counts of Ping Pong Sizes UDP',
    'NTP Ping Pong Sizes': 'Value Counts of Ping Pong Sizes NTP',
}
dictGroups = {
    'External Port':[
        'External Port Dict',
        'External Port TLSTCP',
        'External Port DNS',
        'External Port UDP',
        'External Port NTP'],
    'Hostname':[
        'Hostname Dict',
        'Hostname TLSTCP',
        'Hostname DNS',
        'Hostname UDP',
        'Hostname NTP'],
    'IP':[
        'IP Dict',
        'IP TLSTCP',
        'IP DNS',
        'IP UDP',
        'IP NTP'],
    'Protocol':[
        'Protocol Dict'],
    'Packet Sizes':[
        'Packet Sizes TLSTCP',
        'Packet Sizes DNS',
        'Packet Sizes UDP',
        'Packet Sizes NTP'],
    'Ping Pong Pairs':[
        'Ping Pong Dict',
        'TLSTCP Ping Pong Sizes',
        'DNS Ping Pong Sizes',
        'UDP Ping Pong Sizes',
        'NTP Ping Pong Sizes'
    ]
}
companyCategories = {
    'Ring General Safety': [
        'Ring Security System'],
    'Nest General Safety': [
        'Nest Guard'],
    'Schlage Safety': [
        'Schlage Lock'],
    'D-Link General Safety': [
        'D-Link Alarm'],
    'Belkin Motion Sensors': [
        'Belkin WeMo Motion Sensor'],
    'Chamberlain Motion Sensors': [
        'Chamberlain myQ Garage Opener'],
    'D-Link Motion Sensors': [
        'D-Link Motion Sensor'],
    'Nest Indoor Comfort Sensor': [
        'Nest Thermostat',
        'NEST Protect smoke alarm'],
    'Ecobee Indoor Comfort Sensor': [
        'Ecobee Thermostat'],
    'Netatmo Indoor Comfort Sensor': [
        'Netatmo weather station'],
    'Honeywell Indoor Comfort Sensor': [
        'Honeywell Thermostat'],
    'Unknown Lights' : [
        'Bulb 1'],
    'TP-Link Lights' : [
        'TP-Link Smart WiFi LED Bulb'],
    'LIFX Lights' : [
        'LIFX Virtual Bulb'],
    'Koogeek Lights' : [
        'Koogeek Lightbulb'],
    'Magichome Lights' : [
        'Magichome Strip'],
    'Xiaomi Lights' : [
        'Xiaomi Strip'],
    'Sengled Lights' : [
        'Sengled Bulb'],
    'Philips Lights' : [
        'Philips Hue Bulb'],
    'Ring Lights': [
        "Ring Light"],
    'Belkin Cameras': [
        'Belkin Netcam'],
    'Amazon Cameras': [
        'Cloudcam'],
    'Chinese Cameras':[
        'Chinese Webcam'],
    'Yi Cameras': [
        'Yi Camera'],
    'Luohe Cameras': [
        'Luohe Spycam'],
    'Zmodo Cameras': [
        'Zmodo Doorbell'],
    'Blink Cameras': [
        'Blink Camera'],
    'Lefun Cameras': [
        'Lefun Cam'],
    'TP-Link Cameras:': [
        'TP-Link Kasa Camera',
        'TP-Link Day Night Cloud camera'],
    'Netatmo Cameras': [
        'Netatmo Welcome'],
    'Nest Cameras': [
        'Nest Cam IQ',
        'Nest Hello Doorbell',
        'Nest Cam'],
    'Dropcam Cameras': [
        'Dropcam'],
    'Piper Cameras': [
        'Piper NV'],
    'D-Link Cameras': [
        'D-Link Camera'],
    'Netgear Cameras': [
        'Netgear Arlo Camera'],
    'Insteon Cameras': [
        'Insteon Camera'],
    'Logitech Cameras': [
        'Logitech Logi Circle'],
    'Ring Cameras': [
        'Ring Doorbell',
        'Ring Doorbell Chime',
        'Ring Doorbell Pro'],
    'Amcrest Cameras': [
        'Amcrest Cam'],
    'Withings Cameras': [
        'Withings Home',
        'Withings Smart Baby Monitor'],
    'Geeni Cameras': [
        'Geeni Doorbell Camera',
        'Geeni Camera',
        'Geeni Doorbell',
        'Geeni Aware Camera'], 
    'August Cameras': [
        'August Doorbell Cam'],
    'Wansview Cameras': [
        'Wansview Cam'],
    'Microseven Camera':[
        'Microseven Camera'],
    'Night Owl Cameras':[
        'Nightowl Doorbell'],
    'Samsung Cameras':[
        'Samsung SmartThings Camera'],
    'Bosiwo Cameras':[
        'Bosiwo Camera'],
    'Xiaomi Camera 2': [
        'Xiaomi Camera 2'],
    'Charger Cameras': [
        'Charger Camera'],
    'Merkury Cameras': [
        'Merkury Camera',
        'Merkury Doorbell'],
    'Wyze Cameras:': [
        'Wyze Camera'],
    'Belkin Switches': [
        'Belkin WeMo Link',
        'Belkin WeMo Switch',
        'Belking WeMo Insight'],
    'TP-Link Switches': [
        'TP-Link WiFi Plug'],
    'D-Link Switches': [
        'D-Link Plug'],
    'Amazon Switches': [
        'Amazon Plug'],
    'Smart Switches': [
        'Smart WiFi Plug'],
    'Geeni Switches': [
        'Geeni Smart Plug'],
    'Belkin Cooking': [
        'Belkin WeMo Crockpot'],
    'Smarter Cooking': [
        'iKettle',
        'Smarter Coffee Machine'],
    'GE Cooking': [
        'Microwave'],
    'Samsung Cooking':[
        'Fridge'],
    'Behmor Cooking':[
        'Brewer'],
    'Anova':[
        'Sousvide'],
    'LG TV': [
        'LG Smart TV'],
    'Amazon TV': [
        'Amazon Fire TV'],
    'Apple TV': [
        'Apple TV'],
    'Roku TV': [
        'Roku TV',
        'Roku 4'],
    'Samsung TV': [
        'Samsung SmartTV'],
    'nVidia TV': [
        'nVidia Shield'],
    'Xiaomi Cleaning': [
        'Xiaomi Cleaner'],
    'Roomba Cleaning': [
        'Roomba'],
    'Sengled Hubs': [
        'Sengled Hub'],
    'Insteon Hubs': [
        'Insteon Hub'],
    'Wink Hubs': [
        'Wink 2 Hub'],
    'Logitech Hubs': [
        'Logitech Harmony Hub'],
    'Philips Hubs': [
        'Philips Hue Bridge',
        'Philips Hue Hub'],
    'Caseta Hubs': [
        'Caseta Wireless Hub'],
    'Samsung Hubs': [
        'Samsung SmartThings Hub'],
    'Google Hubs': [
        'Google OnHub'],
    'MiCasaVerde Hubs': [
        'MiCasaVerde VeraLite'],
    'Lightify Hub': [
        'Lightify Hub'],
    'Xiaomi Hub': [
        'Xiaomi Hub'],
    'Arlo Hubs': [
        'Arlo Base Station'],
    'Ultraloq Hubs': [
        'Ultraloq Lock Bridge'],
    'August Hubs': [
        'August Lock Hub',
        'August Hub'],
    'Lockly Hubs': [
        'Lockly Lock Hub'],
    'Sifely Hubs': [
        'Sifely Lock Hub'],
    'Ring Hubs': [
        'Ring Base Station'],
    'Blink Hubs': [
        'Blink Security Hub',
        'Blink Hub'],
    'Simplisafe Hubs': [
        'Simplisafe Hub'],
    'Apple VA': [
        'Apple HomePod'],
    'Sonos VA': [
        'Sonos'],
    'Google VA': [
        'Google Home',
        'Google Home Mini'],
    'Canary VA': [
        'Canary'],
    'Amazon VA': [
        'Amazon Echo',
        'Amazon Echo Dot',
        'Amazon Dot Kids',
        'Amazon Echo Plus',
        'Amazon Echo Show',
        'Amazon Echo Spot',
        'Amazon Echo Look'],
    'iHome VA':[
        'iHome'],
    'Bose VA':[
        'Bose SoundTouch 10'],
    'Harmon Kardon VA': [
        'Allure Speaker',
        'Harmon Kardon Invoke'],
    'Triby VA':[
        'Triby Speaker'],
    'Securifi Router': [
        'Securifi Almond'],
    'Pix-Star PF': [
        'Pix-Star Photo Frame'],
    'Withings Health': [
        'Withings Smart scale',
        'Withings Aura smart sleep sensor'],
    'Rachio Garden': [
        'Rachio Sprinkler',
    ],
    'Simpli Safe Camera': [
        'Simpli Safe'
    ],
    "Foscam Camera": [
        "Foscam"
    ],
    'NoN-IoT': [
        'NoN-IoT'],
}
generalCategories = {
    'General Safety': [
        'Nest Guard',
        'Schlage Lock',
        'D-Link Alarm',
        'Ring Security System'],
    'Motion Sensors': [
        'Belkin WeMo Motion Sensor',
        'Chamberlain myQ Garage Opener',
        'D-Link Motion Sensor'],
    'Indoor Comfort Sensor': [
        'Nest Thermostat',
        'Netatmo weather station',
        'NEST Protect smoke alarm',
        'Ecobee Thermostat',
        'Honeywell Thermostat'],
    'Light' : [
        'Bulb 1',
        'TP-Link Smart WiFi LED Bulb',
        'LIFX Virtual Bulb',
        'Koogeek Lightbulb',
        'Magichome Strip',
        'Xiaomi Strip',
        'Sengled Bulb',
        'Philips Hue Bulb',
        "Ring Light"],
    'Cameras': [
        'Belkin Netcam',
        'Cloudcam',
        'Chinese Webcam',
        'Yi Camera',
        'Luohe Spycam',
        'Zmodo Doorbell',
        'Blink Camera',
        'Lefun Cam',
        'TP-Link Day Night Cloud camera',
        'Netatmo Welcome',
        'Nest Cam IQ',
        'Dropcam',
        'Nest Cam',
        'Piper NV',
        'D-Link Camera',
        'Netgear Arlo Camera',
        'Insteon Camera',
        'Logitech Logi Circle',
        'Ring Doorbell',
        'Ring Doorbell Pro',
        'Amcrest Cam',
        'Withings Home',
        'Withings Smart Baby Monitor',
        'August Doorbell Cam',
        'Wansview Cam',
        'Microseven Camera',
        'Geeni Doorbell Camera',
        'Geeni Camera',
        'Nest Hello Doorbell',
        'Samsung SmartThings Camera',
        'Ring Doorbell Chime',
        'TP-Link Kasa Camera',
        'Bosiwo Camera',
        'Xiaomi Camera 2',
        'Charger Camera',
        "Merkury Camera",
        "Wyze Camera",
        "Nightowl Doorbell",
        "Geeni Doorbell",
        "Merkury Doorbell",
        "Geeni Aware Camera",
        'Simpli Safe',
        "Foscam"],
    'Switches': [
        'TP-Link WiFi Plug',
        'Belkin WeMo Link',
        'Belkin WeMo Switch',
        'Belking WeMo Insight',
        'D-Link Plug',
        'Amazon Plug',
        'Smart WiFi Plug',
        "Geeni Smart Plug"],
    'Cooking': [
        'Belkin WeMo Crockpot',
        'iKettle',
        'Microwave',
        'Fridge',
        'Brewer',
        'Sousvide',
        'Smarter Coffee Machine'],
    'Smart TV': [
        'LG Smart TV',
        'Amazon Fire TV',
        'Apple TV',
        'Roku TV',
        'Roku 4',
        'Samsung SmartTV',
        'nVidia Shield'],
    'Cleaning': [
        'Xiaomi Cleaner',
        'Roomba'],
    'Smart Hubs': [
        'Sengled Hub',
        'Blink Security Hub',
        'Insteon Hub',
        'Wink 2 Hub',
        'Logitech Harmony Hub',
        'Philips Hue Hub',
        'Caseta Wireless Hub',
        'Samsung SmartThings Hub',
        'Google OnHub',
        'MiCasaVerde VeraLite',
        'Lightify Hub',
        'Xiaomi Hub',
        'Arlo Base Station',
        'Ultraloq Lock Bridge',
        'August Lock Hub',
        'Lockly Lock Hub',
        'Sifely Lock Hub',
        'Ring Base Station',
        'Philips Hue Bridge',
        "August Hub",
        "Simplisafe Hub"],
    'Voice Assistants': [
        'Allure Speaker',
        'Apple HomePod',
        'Sonos',
        'Google Home',
        'Google Home Mini',
        'Canary',
        'Amazon Echo',
        'Amazon Echo Dot',
        'Amazon Dot Kids',
        'Amazon Echo Plus',
        'Amazon Echo Show',
        'Amazon Echo Spot',
        'Amazon Echo Look',
        'iHome',
        'Bose SoundTouch 10',
        'Harmon Kardon Invoke',
        'Triby Speaker'],
    'Smart Router': [
        'Securifi Almond'],
    'Photo Frames': [
        'Pix-Star Photo Frame'],
    'Health': [
        'Withings Smart scale',
        'Withings Aura smart sleep sensor'],
    'Garden': [
        'Rachio Sprinkler',
    ],
    'NoN-IoT': [
        'NoN-IoT'],
}
companyDevices = {
    'Ring': [
        'Ring Security System',
        'Ring Light',
        'Ring Doorbell',
        'Ring Doorbell Chime',
        'Ring Doorbell Pro',
        'Ring Base Station'],
    'Nest': [
        'Nest Guard',
        'Nest Thermostat',
        'NEST Protect smoke alarm',
        'Nest Cam IQ',
        'Nest Hello Doorbell',
        'Nest Cam'],
    'Schlage': [
        'Schlage Lock'],
    'D-Link': [
        'D-Link Alarm',
        'D-Link Motion Sensor',
        'D-Link Camera',
        'D-Link Plug'],
    'Belkin': [
        'Belkin WeMo Motion Sensor',
        'Belkin Netcam',
        'Blink Camera',
        'Blink Security Hub',
        'Blink Hub',
        'Belkin WeMo Link',
        'Belkin WeMo Switch',
        'Belking WeMo Insight',
        'Belkin WeMo Crockpot'],
    'Chamberlain Motion Sensors': [
        'Chamberlain myQ Garage Opener'],
    'Geeni': [
        'Geeni Doorbell Camera',
        'Geeni Camera',
        'Geeni Doorbell',
        'Geeni Aware Camera',
        'Geeni Smart Plug'], 
    'Ecobee': [
        'Ecobee Thermostat'],
    'Netatmo': [
        'Netatmo weather station',
        'Netatmo Welcome'],
    'Honeywell': [
        'Honeywell Thermostat'],
    'Unknown 1' : [
        'Bulb 1'],
    'TP-Link' : [
        'TP-Link Smart WiFi LED Bulb',
        'TP-Link Kasa Camera',
        'TP-Link Day Night Cloud camera',
        'TP-Link WiFi Plug'],
    'Google': [
        'Google Home',
        'Google Home Mini',
        'Google OnHub'],
    'LIFX' : [
        'LIFX Virtual Bulb'],
    'Koogeek' : [
        'Koogeek Lightbulb'],
    'Magichome' : [
        'Magichome Strip'],
    'Xiaomi' : [
        'Xiaomi Strip',
        'Xiaomi Camera 2',
        'Xiaomi Cleaner',
        'Xiaomi Hub'],
    'Sengled' : [
        'Sengled Bulb',
        'Sengled Hub'],
    'Amazon': [
        'Cloudcam',
        'Amazon Plug',
        'Amazon Echo',
        'Amazon Echo Dot',
        'Amazon Dot Kids',
        'Amazon Echo Plus',
        'Amazon Echo Show',
        'Amazon Echo Spot',
        'Amazon Echo Look',
        'Amazon Fire TV'],
    'Chinese':[
        'Chinese Webcam'],
    'Yi': [
        'Yi Camera'],
    'Luohe': [
        'Luohe Spycam'],
    'Zmodo': [
        'Zmodo Doorbell'],
    'Lefun': [
        'Lefun Cam'],
    'Dropcam': [
        'Dropcam'],
    'Piper': [
        'Piper NV'],
    'Netgear': [
        'Netgear Arlo Camera',
        'Arlo Base Station'],
    'Insteon': [
        'Insteon Camera',
        'Insteon Hub'],
    'Logitech': [
        'Logitech Logi Circle',
        'Logitech Harmony Hub'],
    'Amcrest Cameras': [
        'Amcrest Cam'],
    'Withings Cameras': [
        'Withings Home',
        'Withings Smart Baby Monitor',
        'Withings Smart scale',
        'Withings Aura smart sleep sensor'],
    'Wansview': [
        'Wansview Cam'],
    'Microseven':[
        'Microseven Camera'],
    'Night Owl':[
        'Nightowl Doorbell'],
    'Bosiwo':[
        'Bosiwo Camera'],
    'Charger': [
        'Charger Camera'],
    'Merkury': [
        'Merkury Camera',
        'Merkury Doorbell'],
    'Wyze': [
        'Wyze Camera'],
    'Gosund': [
        'Smart WiFi Plug'],
    'Smarter Cooking': [
        'iKettle',
        'Smarter Coffee Machine'],
    'GE': [
        'Microwave'],
    'Samsung':[
        'Fridge',
        'Samsung SmartThings Camera',
        'Samsung SmartTV',
        'Samsung SmartThings Hub'],
    'Behmor':[
        'Brewer'],
    'Anova':[
        'Sousvide'],
    'LG': [
        'LG Smart TV'],
    'Apple': [
        'Apple TV',
        'Apple HomePod'],
    'Roku': [
        'Roku TV',
        'Roku 4'],
    'nVidia': [
        'nVidia Shield'],
    'Roomba': [
        'Roomba'],
    'Wink': [
        'Wink 2 Hub'],
    'Philips': [
        'Philips Hue Bridge',
        'Philips Hue Hub',
        'Philips Hue Bulb'],
    'Caseta': [
        'Caseta Wireless Hub'],
    'MiCasaVerde': [
        'MiCasaVerde VeraLite'],
    'Lightify': [
        'Lightify Hub'],
    'Ultraloq': [
        'Ultraloq Lock Bridge'],
    'August': [
        'August Lock Hub',
        'August Hub',
        'August Doorbell Cam'],
    'Lockly': [
        'Lockly Lock Hub'],
    'Sifely': [
        'Sifely Lock Hub'],
    'Simplisafe': [
        'Simplisafe Hub'],
    'Sonos': [
        'Sonos'],
    'Canary': [
        'Canary'],
    'iHome':[
        'iHome'],
    'Bose':[
        'Bose SoundTouch 10'],
    'Harmon Kardon': [
        'Allure Speaker',
        'Harmon Kardon Invoke'],
    'Triby':[
        'Triby Speaker'],
    'Securifi Router': [
        'Securifi Almond'],
    'Pix-Star': [
        'Pix-Star Photo Frame'],
    'Rachio': [
        'Rachio Sprinkler'],
    'NoN-IoT': [
        'NoN-IoT'],
}
renameDevices = {
    "Amazon Echo Dot": ["Amazon Echo Dot-1", "Amazon Echo Dot-2"],
    "LIFX Virtual Bulb": ["LIFX Virtual Bulb-1", "LIFX Virtual Bulb-2"],
    'Simpli Safe': ['Simpli Safe-1', 'Simpli Safe-2'],
    'Nest Cam': ['Nest Cam-1', 'Nest Cam-2'],
    'Geeni Camera': ['Geeni Camera-1', 'Geeni Camera-2', 'Geeni Camera-3'],
    'Ring Doorbell': ['Ring Doorbell-2', 'Ring Doorbell-3'],
    'Ring Doorbell Pro': ['Ring Doorbell Pro-1'],
    'Roku TV': ['Roku TV-1', 'Roku TV-2'],
    'Blink Camera': ['Blink Camera-1', 'Blink Camera-2', 'Blink Camera-3'],
    "August Hub": ["August Hub-1", "August Hub-2"],
    "Geeni Doorbell": ["Geeni Doorbell-1", "Geeni Doorbell-2"],
    "Geeni Aware Camera" : ["Geeni Aware Camera-1", "Geeni Aware Camera-2"],
    "Nightowl Doorbell": ["Nightowl Doorbell-1", "Nightowl Doorbell-2"]
}
permanentRename = {
    'Simpli Safe-1': ['Simplisafe Dev (1)'],
    'Simpli Safe-2': ['Simplisafe Dev (2)'],
    'Geeni Camera-1': ['Geeni Camera (1)'],
    'Geeni Camera-2': ['Geeni Camera (2)'],
    'Geeni Camera-3': ['Geeni Camera (3)'],
    'Ring Doorbell-2': ['Ring Doorbell (2)'],
    'Ring Doorbell-3': ['Ring Doorbell (3)'],
    'Blink Camera-1': ['Blink Camera (1)'],
    'Blink Camera-2': ['Blink Camera (2)'],
    'Blink Camera-3': ['Blink Camera (3)'],
    'Roku TV-1': ['Roku TV (Left)'],
    'Roku TV-2': ['RokuTV (Right)'],
    'Nest Cam-1': ['Nest Cam 1'],
    'Nest Cam-2': ['Nest Cam 2'],
    'Ring Doorbell Pro': ['Ring Doorbell Pro (1)'],
    'Amazon Echo Dot-1': ['Amazon Echo Dot 1'],
    'Amazon Echo Dot-2': ['Amazon Echo Dot 2'],
    "LIFX Virtual Bulb-1": ['LiFX Bulb 1'],
    "LIFX Virtual Bulb-2": ['LiFX Bulb 2'],
    "Amazon Echo Dot": ["Amazon Alexa Dot"],
    "Amazon Echo Show": ["Amazon Alexa Show"],
    "Geeni Doorbell-2": ["Geeni Doorbell (2)"],
    "Geeni Doorbell-1": ["Geeni Doorbell (1)"],
    "August Hub-1": ["August Hub (1)"],
    "August Hub-2": ["August Hub (2)"],
    "Blink Security Hub": ["Blink Hub"],
    "August Hub": ["August Lock Hub"],
    'Pix-Star Photo Frame': ['PIX-STAR Photo-frame'],
    'Google Home': ['Google Home Speaker'],
    'Samsung SmartThings Camera': ['Smartthings Camera', 'Samsung SmartCam'],
    'Philips Hue Bridge': ['Phillips Hue Light Bridge'],
    'Philips Hue Hub': ['Philips HUE Hub', 'T Philips Hub'],
    "Philips Hue Bulb": ["Philips Bulb"],
    'D-Link Camera': ['D-Link DCS-5009L Camera'],
    'Belkin WeMo Crockpot': ['WeMo Crockpot'],
    'Nest Cam': ['Nestcam', 'Nest Camera'],
    "Amazon Echo Plus": ['Amazon Echo Plus 2nd Gen'],
    "Apple TV": ["Apple TV (4th Gen)"],
    "LG Smart TV": ['LG TV'],
    "Belkin WeMo Motion Sensor": ["Belkin wemo motion sensor"],
    "Samsung SmartThings Hub": ["Smart Things", 'Smartthings Hub', 'Samsung SmartThings Hub 2'],
    "LIFX Virtual Bulb": ["Light Bulbs LiFX Smart Bulb"],
    "Belkin WeMo Switch": ["Belkin Wemo switch", "Belking WeMo Plug", 'T Wemo Plug', "Belkin WeMo Plug"],
    "D-Link Motion Sensor": ["Dlink Mov"],
    "Netgear Arlo Camera": ["Netgear Arlo"],
    "TP-Link Smart WiFi LED Bulb": ["TP-Link Bulb"],
    "Wink 2 Hub": ["Wink Hub 2", "Wink Hub"],
    "TP-Link WiFi Plug": ["TP-Link Smart plug", "TP-Link Plug"],
    "Harmon Kardon Invoke": ["Invoke"],
    "Geeni Aware Camera-1": ["Geeni Aware Camera (1)"],
    "Geeni Aware Camera-2": ["Geeni Aware Camera (2)"],
    "Nightowl Doorbell": ["Night Owl Doorbell Camera"],
    "Nightowl Doorbell-1": ["Nightowl Doorbell (1)"],
    "Nightowl Doorbell-2": ["Nightowl Doorbell (2)"],
    "Amcrest Cam": ["Amcrest Camera"]
}
