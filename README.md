# BOE_323258
Python code used for micro PIV analysis for the "3D + time blood flow mapping using SPIM-microPIV in the developing zebrafish heart " paper.
This is a mere modification of the code used in OpenPIV https://github.com/OpenPIV

corr_average_modification_main_file.py contains the modified OpenPIV code for
the zebrafish heart microPIV analysis of frames with z and phase information.

loading_and_piv_funcs_new.py is a "helper" module for the above script.

NOTE: since the first vesrion of the manuscript, the PIV analysis now
adds an extra "padding" of 0 values around the raw data, to allow
for insertion of an extra interrogation window, which allows 
to investigate flow features near the boundaries of the region
of interest. The padding is half the large IW dimension.
