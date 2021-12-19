#!/usr/bin/env python3

# Copyright 2021 Rajbir Bhattacharjee
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import pathlib
import pickle
import json

import threading
import glob

class ChunkDataCommon():
    def __init__(self, directory:str, chunk_size=16):
        self.directory = directory
        self.chunk_size = chunk_size

    def get_pickle_file_name(self, st, en):
        return f"{self.directory}/array_chunk-{st:08d}-{en:08d}.pickle"

    def get_conf_file_name(self):
        with self.lock:
            return f"{self.directory}/config.json"

    def get_slice_start_for_index(self, ind):
        return (ind // self.chunk_size) * self.chunk_size

    def load_slice(self, ind_start):
        with self.lock:
            ind_end = ind_start + self.chunk_size - 1
            filename =self.get_pickle_file_name(ind_start, ind_end)
            with open(filename, "rb") as f:
                self.current_slice = pickle.load(f)
                self.current_slice_start = ind_start
                return
            self.current_slice = None
            self.current_slice_start = -1
            raise IndexError


class ChunkDataWriter(ChunkDataCommon):
    """Uses pickle, but splits into multiple files.
    behaves like an array and supports the append() method,
    and finalize method()"""
    def __init__(self, directory:str, chunk_size=16):
        super(self.__class__, self).__init__(directory, chunk_size)
        self.lock = threading.RLock()

        # The reader object for __getitem__
        self.reader = None

        # the current slice of the array
        self.current_slice = list()

        # Length of all items
        self.length = 0

        # Where does the current slice start from
        self.current_slice_start = -1

        self.__setitem__ = None

        if not os.path.exists(self.directory):
            pathlib.Path(self.directory).mkdir(parents=True, exist_ok=False)

        if os.path.exists(self.get_conf_file_name()):
            self.load_existing()
        else:
            self.write_conf()

    def load_existing(self):
        with open(self.get_conf_file_name(), "r") as f:
            d = json.load(f)
            self.length = d['length']
            self.length = self.length if self.length > 0 else 0
            if self.length > 0:
                ind_start = self.get_slice_start_for_index(self.length - 1)
                self.load_slice(ind_start)
                self.current_slice_start = ind_start

    def finalize(self):
        with self.lock:
            self.write_conf()

    def get_current_slice_max(self):
        # Return the index of the last element that can
        # be stored in the current slice
        with self.lock:
            if self.current_slice_start == -1:
                return -1
            else:
                return self.current_slice_start + self.chunk_size - 1

    def get_current_slice_end(self):
        # Return the end of the current slice
        with self.lock:
            if self.current_slice_start == -1:
                return -1
            else:
                return self.current_slice_start + len(self.current_slice)

    def get_current_slice_size(self):
        # return the size of the current slice
        with self.lock:
            return len(self.current_slice)

    def get_current_slice_offsets(self):
        with self.lock:
            return self.current_slice_start, \
                self.get_current_slice_end(), \
                self.get_current_slice_max()

    def get_current_file_name(self):
        with self.lock:
            if -1 == self.current_slice_start:
                return None
            st = self.current_slice_start
            en = st + self.chunk_size - 1
            return self.get_pickle_file_name(st, en)

    def write_chunk(self):
        with self.lock:
            with open(self.get_current_file_name(), "wb") as f:
                pickle.dump(self.current_slice, f, pickle.HIGHEST_PROTOCOL)


    def write_conf(self):
        with self.lock:
            thedict = {}
            thedict["length"] = self.length
            #print("Exists?", os.path.exists(self.directory))
            with open(self.get_conf_file_name(), "w") as f:
                json.dump(thedict, f)

    def append(self, x):
        with self.lock:
            if 0 == len(self.current_slice) % self.chunk_size:
                self.current_slice = list()
                if self.current_slice_start != -1:
                    self.current_slice_start += self.chunk_size
            if self.current_slice_start == -1:
                self.current_slice_start = 0
            self.length += 1
            self.current_slice.append(x)
            self.write_chunk()
            self.write_conf()
        pass

    def __del__(self):
        with self.lock:
            try:
                # Directory may have been removed by now
                # because del might be called quite late
                if os.path.exists(self.directory):
                    self.write_conf()
            except Exception as e:
                traceback.print_tb(e.__traceback__, file=sys.stderr)

    def __len__(self):
        with self.lock:
            return self.length if self.length >= 0 else 0

    def __getitem__(self, ind):
        with self.lock:
            try:
                # print(self.length, self.directory)
                if self.reader is None or self.reader.length <= ind:
                    self.reader = ChunkDataReader(self, lock=self.lock)
            except Exception as e:
                # print(self.length, self.directory)
                traceback.print_tb(e.__traceback__, file=sys.stderr)
                raise IndexError
            return self.reader[ind]




class ChunkDataReader(ChunkDataCommon):
    def __init__(self, d, chunk_size=16, lock=None):
        if isinstance(d, str):
            directory = d
        elif isinstance(d, ChunkDataReader) or isinstance(d, ChunkDataWriter):
            directory = d.directory
            chunk_size = d.chunk_size

        super(self.__class__, self).__init__(directory, chunk_size=chunk_size)

        self.lock = threading.RLock() if lock is None else lock
        self.length = -1

        self.current_slice = None
        self.current_slice_start = -1
        self.chunk_size = chunk_size
        self.directory = directory

        with open(self.get_conf_file_name(), "r") as f:
            conf = json.load(f)
            length = conf['length']
            self.length = length if length > 0 else -1

    def __len__(self):
        with self.lock:
            return self.length if self.length > 0 else 0


    def __getitem__(self, ind):
        with self.lock:
            if ind >= self.length:
                raise IndexError
            if (ind < 0):
                ind = self.length + ind 
            with self.lock:
                ind_start = self.get_slice_start_for_index(ind)
                if ind_start != self.current_slice_start:
                    self.load_slice(ind_start)
                ind = ind - self.current_slice_start
                return self.current_slice[ind]



