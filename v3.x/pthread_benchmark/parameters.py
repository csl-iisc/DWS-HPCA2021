parameters = {}
# Default values for config variables.
parameters['gpgpu_n_clusters']        =  '30'
parameters['gpgpu_n_mem']             =  '16'
parameters['gpgpu_cache:dl2']         =  '64:128:16,L:T:f:W,A:128:4,4'
parameters['rop_latency']             =  '10'
parameters['fixed_latency_enabled']   =  '0'
parameters['dram_latency']            =  '100'
parameters['tlb_size']                =  '32'
parameters['l2_tlb_entries']          =  '1024'
parameters['l2_tlb_ways']             =  '16'
parameters['l2_tlb_ports']            =  '4'
parameters['l2_tlb_latency']          =  '10'
parameters['pw_cache_enable']         =  '1'
parameters['pw_cache_num_ports']      =  '4'
parameters['pw_cache_latency']        =  '10'
parameters['tlb_pw_cache_entries']    =  '64'
parameters['tlb_pw_cache_ways']       =  '8'
parameters['tlb_bypass_enabled']      =  '0'
parameters['tlb_bypass_level']        =  '1'
parameters['page_walk_queue_size']    =  '192'
parameters['concurrent_page_walks']   =  '16'
parameters['vm_config']               =  '0'
parameters['dwsp_queue_threshold']    =  '0.3'
parameters['dwsp_occupancy_threshold']=  '0.3'

parameters_to_change = {}
# The combinations which need to be generated.
# These names will appear in the folder name.
# parameters_to_change['page_walk_queue_size']   =  [
#         '192',
#         # '32', '192', '1024'
#         ]
# parameters_to_change['gpgpu_cache:dl2']        =  [
        # '32:128:16,L:T:f:W,A:128:4,4',
 #       '64:128:16,L:T:f:W,A:128:4,4',
        # '128:128:16,L:T:f:W,A:128:4,4'
#        ]
# parameters_to_change['concurrent_page_walks']  =  [
#         # '16',
#         '12', '16', '24'
#         ]
# parameters_to_change['tlb_pw_cache_entries']   =  [
#         '64',
#         # '32', '64', '128'
#         ]
# parameters_to_change['l2_tlb_entries']         =  [
#         '512',
#         '256', '512', '1024'
#         ]
parameters_to_change['vm_config']         =  [
        '0'
        # '2', '3', '4', '5', '6', '7'
        # '6'
        ]

# Get these values from the output files.
parameters_to_observe = [
        'gpu_ipc_1',
        'gpu_ipc_2',
        # 'l2 tlb accesses',
        # 'page walks avg latency',
        # 'l2 tlb accesses',
        # 'l2 tlb hits',
        # 'l2 tlb misses',
        # 'l2 tlb mshr hits',
        # 'l2 tlb mshr fails',
        # 'page walks total num',
        # 'page walks avg per app latency',
        # 'page walk app avg waiting time in queue',
        # 'total number of mf',
        # 'averagemflatency',
        # 'total TLBmf',
        # 'averageTLBmflatency',
        # 'totalpage walk returning',
        # 'average page walk latency',
        # 'number of page walk app 0',
        # 'average page walk latency of app 0',
        # 'averageTLBmflatency_0',
        # 'total datamf',
        # 'averagedatamflatency',
        # 'number of data app 0',
        # 'average data latency of app 0',
        # 'tlb_access',
        # 'tlb_hit',
        # 'tlb_miss',
        # 'tlb_mshr',
        # 'gpgpu_n_tlb_misses',
        # 'gpgpu_n_tlb_hits',
        # 'gpgpu_n_tlb_mshr_hits',
	# 'gpgpu_n_tlb_hit_l1cache_reservation_fails_app0'
        ]

# benchmarks = ['BLK', 'LPS', 'MM', '3DS', 'HISTO', 'HS']
# benchmarks = ['SAD', 'SPMV', 'LPS', 'MM', '3DS', 'HISTO', 'HS']
# benchmarks = ['BFS2', 'BP', 'CFD', 'CONS', 'FFT', 'FWT', 'GUPS', 'JPEG', 'LIB',
#         'LPS', 'LUD', 'LUH', 'NN', 'NW', 'QTC', 'RAY', 'RED', 'SAD', 'SC',
#         'SCAN', 'SCP', 'SPMV', 'SRAD', 'TRD', '3DS', 'MM']
benchmarks_a = ['BLK', 'QTC', 'SC', 'SAD', 'GUPS']
benchmarks_b = ['MM', 'HS', 'JPEG', '3DS', 'SRAD']
microBenchmarks = ['MICRO']

pairsBenchmarks = []
for aBenchmark in ['SRAD']:
    for bBenchmark in benchmarks_b:
        pairsBenchmarks.append(aBenchmark + '.' + bBenchmark)
pairsMicroBenchmarks = []
for aBenchmark in microBenchmarks:
    for bBenchmark in microBenchmarks:
        pairsMicroBenchmarks.append(aBenchmark + '.' + bBenchmark)

# benchmarks = microBenchmarks
# benchmarks = benchmarks
# benchmarks = ['BLK']
# benchmarks = pairsMicroBenchmarks
benchmarks = pairsBenchmarks
