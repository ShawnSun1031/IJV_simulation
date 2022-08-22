python3 S2_make_simfile.py "KB_ijv_small_to_large" 300000000
python3 S3_run_sim.py "KB_ijv_small_to_large" 1 765
python3 S4_wmc_generate_dataset.py "KB_dataset_small" "KB_ijv_small_to_large" 1 765
