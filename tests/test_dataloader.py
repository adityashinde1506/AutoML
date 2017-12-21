import unittest
import sys
import logging

#logging.basicConfig(level=logging.INFO)
sys.path.append("/home/adityas/Projects/AutoML")

from mltools.dataset.dataloader import *

class TestDirLoader(unittest.TestCase):

    def setUp(self):
        self.path="/home/adityas/Projects/Experiment_Results"

    def test_file_getter(self):
        files=get_data_files(self.path,["ztomatrix.txt"])
        self.assertIsNotNone(files)

    def test_grouper(self):
        files=get_data_files(self.path,["ztomatrix.txt"])
        grouper=group_filenames(["Aggregation","Collection","Broadcast","Consensus","DGD"])
        data_groups=grouper(files)
        self.assertEqual(len(data_groups),6)

    def test_file_splitter(self):
        files=get_data_files(self.path,["ztomatrix.txt"])
        grouper=group_filenames(["Aggregation","Collection","Broadcast","Consensus","DGD"])
        data_groups=grouper(files)
        dataset=split_files_into_datasets(data_groups) 
        print(dataset)
