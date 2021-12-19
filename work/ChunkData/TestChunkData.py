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
import unittest
import shutil
import pathlib
import glob
import json
import random
from tqdm import tqdm

from ChunkData import *

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

    def test_config_should_have_correct_length(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
            directory=TestChunkDataWriter.thepath,\
            chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(TestChunkDataWriter.chunk_size + 1):
            c.append(i)
            #print(i, len(c.current_slice), c.get_current_file_name())
        with open(f"{TestChunkDataWriter.thepath}/config.json", "r") as f:
            conf = json.load(f)
        self.assertEqual(conf['length'], TestChunkDataWriter.chunk_size + 1,\
            "config has the wrong length")

    def generic_tests(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
            directory=TestChunkDataWriter.thepath,\
            chunk_size=TestChunkDataWriter.chunk_size)
        self.assertEqual(c.get_slice_start_for_index(0), 0)
        self.assertEqual(c.get_slice_start_for_index(1), 1)
        self.assertEqual(\
            c.get_slice_start_for_index(TestChunkDataWriter.chunk_size),\
            TestChunkDataWriter.chunk_size)

    def test_reader_reports_correct_length(self):
        theparent   = TestChunkDataWriter.thepathparent
        thepath     = TestChunkDataWriter.thepath
        chunk_size  = TestChunkDataWriter.chunk_size

        def insert(length, added):
            if os.path.exists(theparent):
                shutil.rmtree(theparent)
            c = ChunkDataWriter(thepath, chunk_size)
            arr = [i for i in range(length)]
            for i in tqdm(arr, "Inserting into array"):
                c.append(i+added)
            self.assertEqual(len(c), length)

        def check_length(length, added):
            c = ChunkDataReader(thepath, chunk_size)
            self.assertEqual(len(c), length, "Length does not match")

        def check_serial_read(length, added):
            c = ChunkDataReader(thepath, chunk_size)
            self.assertEqual(c[0], added, "Failed to fetch array[0]")
            for i in tqdm(range(length), "checking all items"):
                self.assertEqual(c[i], i + added, f"element {i} does not match")

        def check_random_read(length, added):
            c = ChunkDataReader(thepath, chunk_size)
            indices = [i for i in range(length)]
            random.shuffle(indices)
            for i in tqdm(indices, "checking all items in random order"):
                self.assertEqual(c[i], i + added, f"element {i} does not match")


        def do_all_checks(length, added):
            insert(length, added)
            check_length(length, added)
            if length > 0:
                check_serial_read(length, added)
            check_random_read(length, added)

        do_all_checks(0, 3)
        do_all_checks(50, 3)
        do_all_checks(chunk_size - 1, 4)
        do_all_checks(chunk_size + 1, 4)
        do_all_checks(chunk_size, 4)
        do_all_checks(chunk_size * 5 - 1, 4)
        do_all_checks(chunk_size * 5 + 1, 4)
        do_all_checks(chunk_size * 5, 4)

    def test_alternate_reader_init(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        c.append(5)
        c = ChunkDataReader(c)
        self.assertEqual(c[0], 5, "Failed to read correct element")

    def test_negative_index(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(11):
            c.append(i)
        c = ChunkDataReader(c)
        self.assertEqual(c[-1], 10, "element at egative index doesn't match")
        self.assertEqual(c[-2], 9, "element at egative index doesn't match")

    def test_getitem_from_writer(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(11):
            c.append(i)
        self.assertEqual(c[-1], 10, "element at egative index doesn't match")
        self.assertEqual(c[-2], 9, "element at egative index doesn't match")

    def test_close_and_reopen_in_write_mode(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        for i in range(11):
            c.append(i)
        c = None
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        self.assertEqual(c.current_slice[0], 0)
        self.assertEqual(c.current_slice[1], 1)


    def test_getitem_for_writer2(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        c.append(0)
        c.append(1)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 1)
        c.append(2)
        self.assertEqual(c[2], 2)
        for i in range(3, 513):
            c.append(i)
        self.assertEqual(c[256], 256)
        self.assertEqual(c[512], 512)

    def test_setitem_should_not_work(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        c.append(0)
        c.append(1)
        exc_raised = False
        try:
            c[0] = 1
        except:
            exc_raised = True
        self.assertTrue(exc_raised, "assignment of element should be rejected")


    def test_array_iteration(self):
        if os.path.exists(TestChunkDataWriter.thepathparent):
            shutil.rmtree(TestChunkDataWriter.thepathparent)
        c = ChunkDataWriter(\
                directory=TestChunkDataWriter.thepath,\
                chunk_size=TestChunkDataWriter.chunk_size)
        arr = [i for i in range(10)]
        for i in arr:
            c.append(i)
        arr2 = [i for i in c]
        self.assertTrue(arr == arr2)
        c = ChunkDataReader(c)
        arr3 = [i for i in c]
        self.assertTrue(arr == arr3)




if "__main__" == __name__:
    unittest.main()
