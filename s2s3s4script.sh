python3 S2_make_simfile.py "ctchen_ijv_large_to_small" 300000000
python3 S3_run_sim.py "ctchen_ijv_large_to_small" 788 788
python3 S3_run_sim.py "ctchen_ijv_large_to_small" 798 798
python3 S3_run_sim.py "ctchen_ijv_large_to_small" 858 858
python3 S4_wmc_generate_dataset.py "ctchen_dataset_large" "ctchen_ijv_large_to_small" 798 798
python3 S4_wmc_generate_dataset.py "ctchen_dataset_large" "ctchen_ijv_large_to_small" 858 858
python3 S4_wmc_generate_dataset.py "ctchen_dataset_large" "ctchen_ijv_large_to_small" 788 788
