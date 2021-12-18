#!/usr/bin/env python3

import os
import shutil
import pathlib

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
        self.max_length = 0

        # Where does the current slice start from
        self.current_slice_start = -1


        if not os.path.exists(self.directory):
            pathlib.Path(self.directory).mkdir(parents=True, exist_ok=False)

    def get_current_slice_max(self):
        # Return the index of the last element that can
        # be stored in the current slice
        if self.current_slice_start == -1:
            return -1
        else:
            return self.current_slice_start + self.chunk_size - 1

    def get_current_slice_end(self):
        if self.current_slice_start == -1:
            return -1
        else:
            return self.current_slice_start + len(self.current_slice)

    def get_current_slice_size(self):
        return len(self.current_slice)

    def get_current_slice_offsets(self):
        return self.current_slice_start, \
            self.get_current_slice_end(), \
            self.get_current_slice_max()

    def append(self, x):
        with self.lock:
            if self.current_slice_start == -1:
                self.current_slice_start = 0

            self.max_length += 1





        pass


