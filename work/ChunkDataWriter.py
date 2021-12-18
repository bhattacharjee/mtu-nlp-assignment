#!/usr/bin/env python3

import os
import shutil
import pathlib
import pickle
import json

import threading

class ChunkDataWriter():
    """Uses pickle, but splits into multiple files.
    behaves like an array and supports the append() method,
    and finalize method()"""
    def __init__(self, directory:str, chunk_size=16):
        self.directory = directory
        self.chunk_size = chunk_size
        self.lock = threading.RLock()

        # the current slice of the array
        self.current_slice = list()

        # Length of all items
        self.length = 0

        # Where does the current slice start from
        self.current_slice_start = -1


        if not os.path.exists(self.directory):
            pathlib.Path(self.directory).mkdir(parents=True, exist_ok=False)

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
            return f"{self.directory}/array_chunk-{st:08d}-{en:08d}.pickle"

    def get_conf_file_name(self):
        with self.lock:
            return f"{self.directory}/config.json"

    def write_chunk(self):
        with self.lock:
            with open(self.get_current_file_name(), "wb") as f:
                pickle.dump(self.current_slice, f, pickle.HIGHEST_PROTOCOL)


    def write_conf(self):
        with self.lock:
            thedict = {}
            thedict["length"] = self.length
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

    def __len__(self):
        return self.length


