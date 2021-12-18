#!/usr/bin/env python3

import os
import shutil
import pathlib

class ChunkDataWriter():
    """Uses pickle, but splits into multiple files.
    behaves like an array and supports the append() method,
    and finalize method()"""
    def __init__(self, directory:str):
        self.directory = directory

        if not os.path.exists(self.directory):
            pathlib.Path(self.directory).mkdir(parents=True, exist_ok=False)



