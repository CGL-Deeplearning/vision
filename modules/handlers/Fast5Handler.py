from FileManager import FileManager
import numpy as np
import h5py

class H5_Handler:
    """
    Handles fast5 files using h5py and numpy
    """
    def __init__(self,path):
        """
        Creates a dictionary containing all read_id to nanopore file_paths
        :param folder path to directory containing fast5 files.
        """
        read_id_list = []
        file_path_list = []
        file_manager = FileManager()
        file_paths = file_manager.get_file_paths_from_directory(path)
        for file in file_paths: #Needs directory of nanopore_signals
            try:
                h5_file = h5py.File(file,'r')
                read_id = str(h5_file['Analyses/Basecall_1D_000/Configuration/general'].attrs['uuid'])
                read_id_list.append(read_id)
                file_path_list.append(file)
            except(IOError):
                print("FAST5 FILE READ ERROR")
                pass
        self.h5_read_id_file_dir_dict = dict(zip(read_id_list,file_path_list))
    def get_all_Event_Data(self,read_id):
        """
        Returns all event data for a given read_id       
        :return: h5_dataset
        """
        try:
            h5_file = h5py.File(self.h5_read_id_file_dir_dict[read_id],'r')
            return h5_file['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        except:
            raise IOError("FAST5 DIRECTORY ERROR")
    def get_Event_Data(self,read_id):
        """
        Navigates to event data of nanopore file inside H5 object,
        returns a list of mean and a list of st.dv for each kmer.
        Kmer length is 5 in the Fast_5 data structure.
        Possible improvements: Research alternative ways to remove duplication of kmers in Fast5 files.
        :return: list of two lists
        """
        try:
            print(self.h5_read_id_file_dir_dict[read_id])
            h5_file = h5py.File(self.h5_read_id_file_dir_dict[read_id],'r')
            event_data = h5_file['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
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
            kmer_event_list.append(mean_list)
            kmer_event_list = [kmer_event_list[0]]*2 + kmer_event_list + [kmer_event_list[-1]]*2 
            return kmer_event_list
        except:
            raise IOError("FAST5 FILE READ ERROR")

 

           
