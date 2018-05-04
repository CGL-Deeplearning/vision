import shutil
from os import listdir, remove, walk
from os.path import isfile, isdir, join

"""
EXAMPLE USAGE:
bed_directory_path = "/Users/saureous/data/bed_alleles_copy"
output_file_path = "output/bed_output/concatenated.bed"

file_manager = FileManager()
file_paths = file_manager.get_file_paths_from_directory(directory_path=bed_directory_path)
file_manager.concatenate_files(file_paths=file_paths, output_file_path=output_file_path)
file_manager.delete_files(file_paths=file_paths)
"""


class FileManager:
    """
    Does simple file operations like concatenation, fetching a list of paths for files in a directory, deleting files
    """
    @staticmethod
    def concatenate_files(file_paths, output_file_path):
        """
        Concatenate files given in list of file paths to a single file
        :param file_paths: List of file path
        :param output_file_path: Output file path name
        :return: None
        """
        with open(output_file_path, 'wb') as out_file:
            for file_path in file_paths:
                with open(file_path, 'rb') as in_file:
                    # 100MB per writing chunk to avoid reading big file into memory.
                    shutil.copyfileobj(in_file, out_file, 1024*1024*100)

    @staticmethod
    def get_file_paths_from_directory(directory_path):
        """
        Returns all paths of files given a directory path
        :param directory_path: Path to the directory
        :return: file_paths: A list of paths of files
        """
        file_paths = [join(directory_path, file) for file in listdir(directory_path) if isfile(join(directory_path, file))]
        return file_paths

    @staticmethod
    def get_subdirectories(directory_path):
        """
        Collect the absolute paths of all top-level directories contained in the specified directory
        :param directory_path: Path to a directory containing subdirectories
        :return: dir_paths: list of paths
        """
        dir_paths = [join(directory_path, file) for file in listdir(directory_path) if isdir(join(directory_path, file))]
        return dir_paths

    @staticmethod
    def get_all_filepaths_by_type(parent_directory_path, file_extension, sort=True):
        all_files = list()

        for root, dirs, files in walk(parent_directory_path):
            sub_files = [join(root,subfile) for subfile in files if subfile.endswith(file_extension)]
            all_files.extend(sub_files)

        all_files.sort()

        return all_files

    @staticmethod
    def delete_files(file_paths):
        """
        Deletes files given in file paths
        :param file_paths: List of file paths
        :return: None
        """
        for file_path in file_paths:
            remove(file_path)

