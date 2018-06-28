import glob
import numpy as np
import h5py

class H5_Handler:
    """
    Handles fast5 files using h5py and numpy
    """
    def __init__(self,read_id):
        """
        Creates a dictionary containing all read_id to nanopore file_paths
        Work needed: change glob import to file manager
        :param read_id
        """
        self.read_id = read_id
        read_id_key = []
        h5_file_path_values = []
        for self.h5_file_path in glob.iglob('/Users/mathadmin/Desktop/Jacob/Nanopore_signal_files/*.fast5'): #Needs directory of nanopore_signals
            try:
                self.h5_file = h5py.File(self.h5_file_path,'r')
            except:
                raise IOError("FAST5 FILE READ ERROR")
            read_id_directory_location = self.h5_file['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq']
            read_id_event_data_numpy = np.array(read_id_directory_location)
            h5_list = str(read_id_event_data_numpy, 'UTF8')
            h5_read_Id = h5_list.split()
            h5_read_Id = h5_read_Id[0][1:]
            h5_read_Id.join('')
            read_id_key.append(h5_read_Id)
            h5_file_path_values.append(self.h5_file_path)
        self.myData = dict(zip(read_id_key, h5_file_path_values))

    def get_Event_Data(self):
        """
        Navigates to event data of nanopore file inside H5 object,
        returns a list of mean and a list of st.dv for each kmer.
        Kmer length is 5 in the Fast_5 data structure.
        Possible improvements: Research alternative ways to remove duplication of kmers in Fast5 files.
        :return: list of two lists
        """
        try:
            h5_file = h5py.File(self.myData[self.read_id],'r')
            event_data = h5_file['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
            event_data = np.array(event_data)
            mean_list = []
            st_dv_list = []
            kmer_event_list = []
            for i in range(len(event_data)):
                if(event_data[i][5] == 0 and i != 0): #removes duplication of kmers
                    pass
                else:
                    mean_list.append(event_data[i][0])
                    st_dv_list.append(event_data[i][2])
            kmer_event_list.append([mean_list,st_dv_list])
            kmer_event_list = [kmer_event_list[0]]*2 + kmer_event_list + [kmer_event_list[-1]]*2 
            return kmer_event_list
        except:
            raise IOError("FAST5 FILE READ ERROR")