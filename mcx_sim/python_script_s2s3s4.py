# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 03:12:12 2022

@author: md703
"""

import os

if __name__ == "__main__":
    # os.system("python S2_make_simfile.py ctchen_cvtest_1e8_ijv_large_to_small 100000000")
    os.system("python S3_run_sim.py ctchen_cvtest_1e8_ijv_large_to_small 1 1225")
    # os.system("python S4_wmc_generate_dataset.py ctchen_dataset_large ctchen_cvtest_1e8_ijv_large_to_small 1 1225")
    # os.system("python S2_make_simfile.py ctchen_cvtest_1e8_ijv_small_to_large 100000000")
    # os.system("python S3_run_sim.py ctchen_cvtest_1e8_ijv_small_to_large 1 1225")
    # os.system("python S4_wmc_generate_dataset.py ctchen_dataset_small ctchen_cvtest_1e8_ijv_small_to_large 1 1225")