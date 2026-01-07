# Test Run 
python bradley_terry.py -it True -ng 100

# Run on progress-as-preference case
python process_roboarena_data_for_bradley_terry.py
python bradley_terry.py -p 0 -dp "Roboarena_BT.npy" -nt 7