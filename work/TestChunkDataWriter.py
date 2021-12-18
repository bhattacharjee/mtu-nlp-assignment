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

        st, end, mx = c.get_current_slice_offsets()
        self.assertEqual(st, 0, "start offset should be zero")
        self.assertEqual(end, 1, "end offset should be 1")
        self.assertEqual(mx, TestChunkDataWriter.chunk_size - 1,\
            "max should be {TestChunkDataWriter.chunk_size - 1}")
        self.assertEqual(c.get_current_slice_size(), 1,\
            "slice size should be 1")

        # Test that a pickle file is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.pickle")
        self.assertEqual(len(file_list), 1, "No pickle file created")

        # Test that a json file describing the item is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.json")
        self.assertEqual(len(file_list), 1, "No json conf created")

    def test_write_as_many_as_chunk_size(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
            directory=TestChunkDataWriter.thepath,\
            chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(TestChunkDataWriter.chunk_size):
            c.append(i)
        # Test that a pickle file is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.pickle")
        self.assertEqual(len(file_list), 1, "pickle file count mismatch")

        # Test that a json file describing the item is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.json")
        self.assertEqual(len(file_list), 1, "No json conf created")

    def test_write_one_more_than_chunk_size(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
            directory=TestChunkDataWriter.thepath,\
            chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(TestChunkDataWriter.chunk_size + 1):
            c.append(i)
            #print(i, len(c.current_slice), c.get_current_file_name())
        # Test that a pickle file is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.pickle")
        self.assertEqual(len(file_list), 2, "pickle file count mismatch")

        # Test that a json file describing the item is created
        file_list = glob.glob(f"{TestChunkDataWriter.thepath}/*.json")
        self.assertEqual(len(file_list), 1, "No json conf created")




if "__main__" == __name__:
    unittest.main()
