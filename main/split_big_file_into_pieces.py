"""
Take a large csv or text file and split it line by line into chunks.
"""
import math
import os


def split_big_file_into_pieces(directory, file, chunk_size):
    """Split a big file into smaller chunks"""
    os.chdir(directory)
    with open(file) as f_in:
        headers = f_in.readline()
        f_out = open("SNF_MDS_0.txt", "w")
        f_out.write(headers)
        for index, line in enumerate(f_in):
            f_out.write(line)
            if(index+1) % chunk_size == 0:
                print(index)
                f_out.close()
                f_out = open("SNF_MDS_" + str(int(math.ceil(index/chunk_size))) + ".txt",
                             "w")
                f_out.write(headers)
        f_out.close()


if __name__ == "__main__":
    split_big_file_into_pieces(".", "sdf.txt", 5000)
