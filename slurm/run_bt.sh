# # Test Run 
# python bradley_terry.py -it 1 -ng 100

# Run on progress-as-preference case
# python process_roboarena_data_for_bradley_terry.py
python bradley_terry.py -p 0 -dp "Roboarena_progress.npy" -nt 7
python bradley_terry.py -p 1 -dp "Roboarena_preference.npy" -nt 7