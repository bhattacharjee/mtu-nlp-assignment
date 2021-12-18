#!/usr/bin/env python3
import os
import unittest
import shutil
import pathlib

from ChunkDataWriter import *

class TestChunkDataWriter(unittest.TestCase):

    # Test if the destination directory is being created
    def test_directory_creation(self):
        thepath = "/tmp/1/2"
        thepathparent = "/tmp/1"
        if os.path.exists(thepathparent):
            shutil.rmtree(thepathparent)
        c = ChunkDataWriter(directory=thepath)
        self.assertTrue(os.path.exists(thepath))
        self.assertTrue(os.path.isdir(thepath))
        shutil.rmtree(thepathparent)


if "__main__" == __name__:
    unittest.main()
