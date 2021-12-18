#!/usr/bin/env python3
import os
import unittest
import shutil
import pathlib
import glob

from ChunkDataWriter import *

class TestChunkDataWriter(unittest.TestCase):
    thepath = "/tmp/1/2"
    thepathparent = "/tmp/1"
    chunk_size = 16

    # Test if the destination directory is being created
    def test_directory_creation(self):
        thepath = TestChunkDataWriter.thepath
        thepathparent = TestChunkDataWriter.thepathparent
        if os.path.exists(thepathparent):
            shutil.rmtree(thepathparent)
        c = ChunkDataWriter(directory=thepath)
        self.assertTrue(os.path.exists(thepath))
        self.assertTrue(os.path.isdir(thepath))
        shutil.rmtree(thepathparent)

    def test_write_single_item(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
            directory=TestChunkDataWriter.thepath,\
            chunk_size=TestChunkDataWriter.chunk_size)
        c.append(1)

        # Test that a pickle file is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.pickle")
        self.assertEqual(len(file_list), 1, "No pickle file created")

        # Test that a json file describing the item is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.json")
        self.assertEqual(len(file_list), 1, "No json file created")

        st, end, mx = c.get_current_slice_offsets()
        assertEqual(st, 0, "start offset should be zero")
        assertEqual(end, 1, "end offset should be 1")
        assertEqual(mx, TestChunkDataWriter.chunk_size - 1,\
            "max should be {TestChunkDataWriter.chunk_size - 1}")




if "__main__" == __name__:
    unittest.main()
