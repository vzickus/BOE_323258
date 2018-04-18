import os, sys
sys.path.insert(0, '/home/vytas/1phd/j_postacquisition_fork/')
from image_loading import *
from loading_and_piv_funcs_new import *
from load_piv_images import sortForPiv
from datetime import datetime

window_height,window_width = 24, 24
window_big_height, window_big_width = 48, 72
# Small IW overlap
overlap_height,overlap_width = 12, 12
#Big IW overlap
overlap_big_height = window_big_height - (window_height - overlap_height)
overlap_big_width = window_big_width - (window_width - overlap_width)
#Centering the IWs
subwindow_inset_height = (window_big_height-window_height)/2
subwindow_inset_width = (window_big_width - window_width)/2
in_bin,fin_bin,bin_size = 0.0, 2*np.pi, 0.209
wkdir = datetime.now().strftime('./corr_avg/%Y-%m-%d %H.%M.%S_1::2_' 
+ str(window_height) + '_' + str(window_width) + '_HW_' + str(window_big_height)
+ '_' +str(window_big_width) +  '_ol_hw_' + str(overlap_height) + '_'
+ str(overlap_width) + '_in_bin_' +str(in_bin) + '_fin_bin_' + str(fin_bin) + '_step_' + str(bin_size) + '/' )
folders = ['flw', 'xyuvms2n', 'corr_avg', 'metadata']
flw_dir,xyuv_dir,corr_avg_dir,meta_dir = [folder_writer(wkdir, folder) for folder in folders]
###PARAMS###
data_type = 'int32'
corr_method =  'sad'#'sad' # 'fft'
mode =  'full' # 'full' # "choose 'full' for same sized windows"
dt = 1.0
px_size = 1.0 #0.65 # effective (i.e. (true pixel size in camera)/magnification) pixel size in microns
subpixel_method = 'parabolic'
sig2noise_method = 'peak2peak'
threshold = 1.0700
width = 3 #half width of the search area/mask around the peak
frame_delta = 1
initial_frame = 0
mean_intensity_thres = 1e5



path_to_bf_plist = r'/home/vytas/1phd/rawdata/fRBC/scan_heart_fRBC/2017-01-11 15.37.20 vid heart piv stack/'
path_to_images = '/home/vytas/1phd/rawdata/fRBC/scan_heart_fRBC/2017-01-11 15.37.20 vid heart piv stack/'
bf_source = 'Allied Vision Technologies GS650 0001f61c'
flr_source = 'Red - QIClick Q35979'

sorted_path = '/home/vytas/1phd/plist parsing/code_for_paper/corr_avg/'



par_vals = dict( (name,eval(name)) for name in ['window_height','window_width',
'subwindow_inset_height','subwindow_inset_width','overlap_height',
'overlap_width','window_big_height','window_big_width','overlap_big_height',
'overlap_big_width','corr_method','mode','dt','px_size','subpixel_method',
'sig2noise_method','threshold',
'width','frame_delta','initial_frame', 'path_to_images', 'mean_intensity_thres', 'in_bin', 'fin_bin', 'bin_size'])

with open(str(wkdir) + '/pars_test.csv', 'wb') as f:
    writer = csv.writer(f,delimiter='=')
    for row in par_vals.iteritems():
        writer.writerow(row)
        
   
nr_planes = 15
initial_plane = 0 
skip_data = None

print range(initial_plane, nr_planes)

for z_counter, z in enumerate(tqdm(range(initial_plane, nr_planes), desc = 'analysing plane z ')):
    print "z counter: ", z_counter
    img_files = np.load(sorted_path + '_PIV_sorted_z_' + str("%03d" %(z+1)) + '.npy') 

    print len(img_files)
    
    
    img_files = img_files[1::2]    
    print img_files[1::2]  
    nr_pairs = len(img_files)
    if nr_pairs < 1:
        print "not enough pairs. Possibly end of z range?"
        continue
    print "First frame in z ", z, "is " , os.path.basename((img_files[z][0].path))

    if nr_pairs > 0:

        #This will yield 6.4 as the final value. Technically anything above 6.28 should go to 0
        #so add a modulo operator.
        ph_bins = np.linspace(0,2*np.pi, 31)#np.arange(in_bin,fin_bin,bin_size)

        for counter, i in enumerate(tqdm(range(len(ph_bins)-1), desc='Running PIV analysis')):
            #write some metadata            
            metaFile = open(meta_dir + "/frame_info_z_" + str("%03d" %z) + "_bin_" + str("%1.3f" % ph_bins[i]) + ".csv" , 'wb')
            writer=csv.writer(metaFile, delimiter = '\t')


            #corr_single_subdir = folder_writer(corr_avg_dir,'/single_corrs_ph_bin_' +cy str("%.1f" % ph_bins[i]))                 
            xyuv_subdir = folder_writer(xyuv_dir,"/bin_" + str("%1.3f" % ph_bins[i]))                
            #sum_corrs_dir = folder_writer(corr_avg_dir,"/cummulative_corrs_ph_bin_" + str("%.1f" % ph_bins[i]) )
            print "bin:", ph_bins[i], ph_bins[i+1]
            phase_matched = img_files[np.where( (ph_bins[i] <= img_files[:,2]) * (img_files[:,2] < ph_bins[i+1]))]

            if len(phase_matched) > 0:
                frame_c = np.zeros(pad_array(scipy.misc.imread(img_files[z][0].path).astype(data_type), [window_big_height/2,window_big_width/2]).shape)
                frame_d = frame_c
            else:
                print "no frames in bin ", i
                continue
            
######################################################################
            writer.writerow(["z", "frame 1", "frame 2", "phase", "nr of pairs"] )
            for counting, frames in enumerate(phase_matched):
                # METADATA
                writer.writerow([z, os.path.basename(frames[0].path)[:-4], os.path.basename(frames[1].path)[:-4], round(frames[2],3), len(phase_matched)] )

                if counting == 0:
                    print "Clearing sum frames"
                    frame_a_sum = np.zeros(pad_array(scipy.misc.imread(img_files[z][0].path).astype(data_type), [window_big_height/2,window_big_width/2]).shape)
                    frame_b_sum = frame_a_sum 
                    frame_ab_sum = frame_a_sum
                frame_a = scipy.misc.imread(frames[0].path).astype(data_type)
                frame_b = scipy.misc.imread(frames[1].path).astype(data_type)
                frame_a = pad_array(frame_a, [window_big_height/2,window_big_width/2]).astype(data_type)
                frame_b = pad_array(frame_b, [window_big_height/2,window_big_width/2]).astype(data_type)  
                frame_a_sum = np.add(frame_a_sum, frame_a)
                frame_b_sum = np.add(frame_b_sum, frame_b)
                frame_ab_sum = np.add(frame_a_sum, frame_b_sum)
            metaFile.close()
            print "Number of frames summed above: ", len(phase_matched)
######################################################################          
            
                                    
            for count_phase_matched_frames, frame in enumerate(phase_matched):
                if len(phase_matched) > 0:
                    corr_dir = str(wkdir + "/corr")
                                        
                    try:
                        frame_a = scipy.misc.imread(frame[0].path).astype(data_type)
                        frame_b = scipy.misc.imread(frame[1].path).astype(data_type)
                        frame_a = pad_array(frame_a, [window_big_height/2,window_big_width/2]).astype(data_type)
                        frame_b = pad_array(frame_b, [window_big_height/2,window_big_width/2]).astype(data_type)  


                        
                        
                        mean_a = np.mean(frame_a)
                        mean_b = np.mean(frame_b)
            
                        frame_c = np.add(frame_c, frame_a)
                        frame_d = np.add(frame_d, frame_b)
                    except:
                        print "no more frames to match, z, i is : ", z, i
                        continue
                else:
                    print "did not find any frames for z, i: ", z,i 
                    pass
                    
                image_size = frame_a.shape
                #size of the "cropped" image
                image_size_small = (image_size[0] - 2*subwindow_inset_height , image_size[1] - 2*subwindow_inset_width)
    
                #Small IW centres.subwindow_inset_width/height are added to compensate for different sizes in IWs.
                x,y = (get_coordinates( image_size_small, window_height, window_width, overlap_height,overlap_width)[0] + subwindow_inset_width, get_coordinates( image_size_small, window_height, window_width, overlap_height,overlap_width )[1] + subwindow_inset_height)
                #Big IW centres
                q,p = (get_coordinates(image_size, window_big_height,window_big_width, overlap_big_height,overlap_big_width)[0], get_coordinates(image_size, window_big_height,window_big_width, overlap_big_height,overlap_big_width)[1])
                #Check if the IW's have matching centres.
                assert(np.array_equal(x, q) == True)
                assert(np.array_equal(y, p) == True)
                windows_a = moving_sub_window_array(frame_a, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width).astype(data_type)        
                windows_b = moving_window_array( frame_b,window_big_height,window_big_width, overlap_big_height,overlap_big_width).astype(data_type)                    
                windows_a_sum = moving_sub_window_array(frame_a_sum, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width).astype(data_type)
                windows_b_sum = moving_window_array( frame_b_sum,window_big_height,window_big_width, overlap_big_height,overlap_big_width).astype(data_type)
                windows_ab_sum = moving_sub_window_array(frame_ab_sum, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width).astype(data_type)
                corr_shape = correlate_windows( windows_a[0], windows_b[0], corr_method = corr_method, mode = mode).shape
                
                if count_phase_matched_frames == 0:
                    print "corr big cleared"
                    corr_big = np.zeros((windows_a.shape[0],corr_shape[0],corr_shape[1]))
                    corr = np.zeros((windows_a.shape[0],corr_shape[0],corr_shape[1]))
                    
                assert(windows_a.shape[0] == windows_b.shape[0])
                n_rows, n_cols = get_field_shape ( frame_a.shape , window_big_height, window_big_width, overlap_big_height,overlap_big_width ) 
                u = np.zeros(n_rows*n_cols).astype(np.float64)
                v = np.zeros(n_rows*n_cols).astype(np.float64)
                sig2noise = np.zeros(n_rows*n_cols).astype(np.float64)
                
                for j in range(windows_a.shape[0]):
                    if (np.sum(windows_a_sum[j])/len(phase_matched)) >= mean_intensity_thres:
                        corr[j] = correlate_windows( windows_a[j], windows_b[j], corr_method = corr_method, mode = mode)
                        assert(corr.dtype == np.float64)
                    else:
                        pass

                #add the single frame pair correlations to a big array which is then the correlation averaged matrix.
                corr_big = np.add(corr_big,corr)
                               
            corr_big = ((corr_big + 0.0) / len(phase_matched)).astype(np.float64)
            #np.save( corr_avg_dir +  "/z_plane_" + str("%03d" %z) +"_ph_bin" + str("%.1f" % ph_bins[i]) + "_nr_pairs_" + str("%03d" % len(phase_matched)), corr_big)
            print "corr big shape: ", corr_big.shape
            
            u = np.zeros(n_rows*n_cols).astype(np.float64)
            v = np.zeros(n_rows*n_cols).astype(np.float64)
            sig2noise = np.zeros(n_rows*n_cols).astype(np.float64)
            #PIV using correlation averaged results
            for k in tqdm(range(corr_big.shape[0]), desc = 'finding subpixel vals for each iw'):    
                row, col = find_subpixel_peak_position( corr_big[k], subpixel_method=subpixel_method, corr_method = corr_method)
                u[k], v[k] = px_size*(col - corr_shape[1]/2), -px_size*(row - corr_shape[0]/2)
                sig2noise[k] = sig2noise_ratio( corr_big[k], sig2noise_method=sig2noise_method,corr_method= corr_method, width=width )
                
            u,v,sig2noise = u.reshape(n_rows, n_cols), v.reshape(n_rows, n_cols),sig2noise.reshape(n_rows, n_cols)
            u,v, mask = sig2noise_val( u, v, sig2noise, threshold)

           
            xyuvms2n        = xyuv_dir + "/xyuvms2n_z_plane_" + str("%03d" %z) + "_bin_" + str("%1.3f" % ph_bins[i]) + "_nr_pairs_" + str("%03d" % len(phase_matched)) +".txt"
            background      = frame_a
            save( x, y, u, v, mask, sig2noise, xyuvms2n, fmt='%8.4f', delimiter='\t' )
            scipy.misc.imsave(flw_dir + "/z_plane_" + str("%03d" %z) + "_bin_" + str("%1.3f" % ph_bins[i]) + "_nr_pairs_" + str("%03d" % len(phase_matched)) + "C.png", frame_c)
            scipy.misc.imsave(flw_dir + "/z_plane_" + str("%03d" %z) + "_bin_" + str("%1.3f" % ph_bins[i]) + "_nr_pairs_" + str("%03d" % len(phase_matched)) + "D.png", frame_d)
