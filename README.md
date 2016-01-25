# cpu-predict
A simple example of predicting system load based on running processes

Aside from the code, this is a typical output

    RMSE 2.64346185825, cuttoff: 0, SCORE: 0.956799215863

RMSE is root mean squared error, with a total F1 score of .95

    syst_direct_ipo_rate|syst_buffered_ipo_rate|app07_bufio|hour|io_logical_name_trans|app05_dirio|llb0_pkts_sentpsec|app05_bufio|io_split_transfer_rate|
    llb0_pkts_recvpsec|page_modified_list_size|io_file_open_rate|app02_dirio|app05_pagesproc|app08_proccount|page_free_list_size|app05_pagesgbl|
    app08_pagesgbl|app08_pagesproc|app02_bufio|state_lef|io_mailbox_write_rate|page_demand_zero_faults|app02_pagesgbl|io_page_writes|syst_process_count|
    app03_pagesproc|page_page_write_ipo_rate|weekday|page_global_valid_fault_rate|app07_pagesproc|app02_pagesproc|state_compute|app04_pagesproc|
    syst_page_fault_rate|page_free_list_faults|tcp_in|app04_dirio|app01_dirio|app07_pagesgbl|io_page_reads|page_modified_list_faults|tcp_out|
    syst_page_read_ipo_rate|app01_bufio|app04_pagesgbl|tcp_rxdup|state_hib|app05_pgflts|app04_bufio|app08_dirio|app03_pagesgbl|syst_other_states|
    app08_bufio|app02_proccount|app07_pgflts|tcp_kpalv|app04_pgflts|app01_pagesproc|app03_pgflts|app07_dirio|app07_proccount|app03_dirio|app03_bufio|
    tcp_retxpk|state_mwait|tcp_retxto|app06_pagesproc|lla0_pkts_recvpsec|app01_pagesgbl|app08_pgflts|app05_proccount|app06_pagesgbl|app04_proccount|
    state_cur|app02_pgflts|app01_pgflts|app01_proccount|ewd0_pkts_recvpsec|ewd0_pkts_sentpsec|ewc0_pkts_sentpsec|ewc0_pkts_recvpsec|lla0_pkts_sentpsec|
    app06_bufio|app06_proccount|app03_proccount|app06_pgflts|app06_dirio
    
These names above are seperated by a '|' and represent names of different features that went into predicting the output 
CPU load of a running system using the [T4 system logger](http://h71000.www7.hp.com/openvms/products/t4/)

I'm confident that a similar approach could be used to help predict different system load given different configurations for
something modern like Nginx.
