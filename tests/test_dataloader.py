import unittest
import sys

sys.path.append("/home/adityas/Projects/AutoML")

from mltools.dataset.dataloader import *

class TestDirLoader(unittest.TestCase):

    def setUp(self):
        self.path="/home/adityas/Projects/Experiment_Results"

    def test_file_getter(self):
        files=get_data_files(self.path,["ztomatrix.txt"])
        self.assertTrue(len(files)>0)
