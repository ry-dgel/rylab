import numpy as np
import re

################
# Lock-In Data #
################
# Figures out the delimiter used in a csv style file.
def get_delim(file):
    with open(file, "r") as f:
        # Reads file line by line, so that only one line needs to be read
        for _, line in enumerate(f):
            # ignore comments and header if present
            if re.search("[a-df-z]", line) is not None or line.strip() == "":
                continue
            
            for delim in [",", ";", ":", "\t"]:
                if delim in line:
                    return delim

    print("No delimiters found")
    return ""

# Determine how many lines of header are in a file.
def get_header(file):
    with open(file, "r") as f:
        i = 0
        for _, line in enumerate(f):
            if re.search("[a-df-z]", line) is not None or line.strip() == "":
                i += 1
            else:
                break
        return i


# Read data from csv file. 
def read_csv(file, names=True, delim=None, head=None):
    if delim is None:
        delim = get_delim(file)
    if head is None:
        head = get_header(file)
    if names:
        return np.genfromtxt(file, names=True, delimiter=delim,skip_header=head-1)
    else:
        return np.genfromtxt(file, delimiter=delim, skip_header=head)

# Get data from csv file exported from lock in.
def unpack(filename, fields = []):
    chunks = {}
    with open(filename) as f:
        # Skip header line
        next(f)
        for line in f:
            # Each line has form:
            # chunk;timestamp;size;fieldname;data0;data1;...;dataN
            delim = get_delim(filename)
            entries = line.split(delim)

            chunk = entries[0]
            # If this is a new chunk, add to chunks dictionary.
            if chunk not in chunks.keys():
                chunks[chunk] = {}
            # Use chunk dictionary for data storage
            # This separates the runs
            dic = chunks[chunk]

            fieldname = entries[3]
            data = np.array([float(x) for x in entries[4:]])

            # Add named dataset to dictionary for each desired fieldname
            # If no fieldnames specified in fields, just return all.
            if fieldname in fields or len(fields) == 0:
                if fieldname not in dic.keys():
                    dic[fieldname] = data
                else:
                    dic[fieldname] = np.concatenate((dic[fieldname], data))

    data_chunks = list(chunks.values())
    return data_chunks
