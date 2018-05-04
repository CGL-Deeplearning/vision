from os import path, mkdir
import datetime


class TsvWriter:
    def __init__(self, output_directory, header, filename_prefix, use_datetime_suffix=False):
        import csv

        self.directory = output_directory

        # ensure output directory exists
        if not path.exists(self.directory):
            mkdir(self.directory)

        # get unique time tag for this run
        t = datetime.datetime
        datetime_string = '-'.join(list(map(str, t.now().timetuple()))[:-1])

        # generate file path for output
        self.filename = filename_prefix
        if use_datetime_suffix:
            self.filename = self.filename + datetime_string
        self.file_path = path.join(self.directory, self.filename)

        # create and initialize file with header
        self.file = open(self.file_path, 'w')
        self.writer = csv.writer(self.file, delimiter='\t')

        self.append_row(header)

    def append_row(self, row):
        self.writer.writerow(row)
        self.file.flush()
