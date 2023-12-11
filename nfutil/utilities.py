# %% ============================================================================
# SOME MORE IMAGE PROCESSING
# ===============================================================================

# Omega generator function
def generate_omega_edges(meta_df,num_imgs_per_scan):
    """ takes in a dataframe generated from the metadata using "skim_metadata",
    and returns a numpy array of the omega data (ie, what frames represent
    which omega angle in the results)"""
    start = meta_df["ome_start_real"].to_numpy() # ome_start_real is the starting omega position of the first good frame's omega slew
    stop = meta_df["ome_end_real"].to_numpy() # ome_end_read is the final omega position of the last good frame's omega slew
    steps = num_imgs_per_scan
    num_frames_anticipated = meta_df['nframes_real'].to_numpy()
    # This *should* not be needed if meta data was written correctly
    if np.sum(num_frames_anticipated-steps) != 0:
        print('Editing omega stop postion from the metadata to deal with fewer images')
        print('CHECK YOUR OMEGA EDGES AND STEP SIZE WITH np.gradient(omega_edges_deg)')
        stop = stop - (stop/num_frames_anticipated)*(num_frames_anticipated-steps)
    scan = meta_df["SCAN_N"].to_numpy() 
    # Find the omega start positions for each frame
    lines = [np.linspace(a, b - (b - a) / c, c)
             for a, b, c in zip(start, stop, steps)]
    omes = np.hstack([x for y, x in sorted(zip(scan, lines))])
    omega_edges_deg = np.append(omes,stop[np.shape(stop)[0]-1])
    return omes, omega_edges_deg

def make_beamstop_mask(raw_image_stack,num_img_for_median,binarization_threshold,errosions,dilations,feature_size_to_remove):
    # Grab the first image
    short_image_stack = raw_image_stack[0:num_img_for_median,:,:]
    # Take the median of this
    img = np.median(short_image_stack, axis=0)
    # Binarize
    binary_img = img>binarization_threshold
    # Fill holes
    working_img = scipy.ndimage.binary_fill_holes(binary_img)
    # Errode and dilate
    working_img = scipy.ndimage.binary_erosion(working_img, iterations=errosions)
    working_img = scipy.ndimage.binary_dilation(working_img, iterations=dilations)
    # Remove any small features
    working_img = skimage.morphology.remove_small_objects(working_img,feature_size_to_remove,connectivity=1)
    # Invert the image
    working_img = working_img == 0
    # Return 
    return working_img

# %% ============================================================================
# CALIBRATION FUNCTION
# ===============================================================================
def calibrate_parameter(experiment,controller,image_stack,calibration_parameters):
    # Which parameter?
    experiment_parameter_index = [3,4,5,0,1,2,6]
    parameter_number = experiment_parameter_index[calibration_parameters[0]] # 0=X, 1=Y, 2=Z, 3=RX, 4=RY, 5=RZ, 6=chi
    # How many iterations
    iterations = calibration_parameters[1]
    # Start and stop points
    start = calibration_parameters[2]
    stop = calibration_parameters[3]
    # Variable name
    names = ['Detector Horizontal Center (X)',
             'Detector Vertical Center (Y)',
             'Detector Distance (Z)',
             'Detector X Rotation (RX)',
             'Detector Y Rotation (RY)',
             'Detector Z Rotation (RZ)',
             'Chi Angle']
    parameter_name = names[calibration_parameters[0]]

    # Copy the original experiment to work with
    working_experiment = copy.deepcopy(experiment)

    # Calculate the test coordinates
    if parameter_number == 4:
        # Testing vertical detector translation
        test_crds_full, n_crds, Xs, Ys, Zs = gen_nf_test_grid_vertical(experiment.cross_sectional_dimensions, experiment.vertical_bounds, experiment.voxel_spacing)
        to_use = np.arange(len(test_crds_full))
        test_coordinates = test_crds_full[to_use, :]
    else:
        # Testing any of the others
        test_crds_full, n_crds, Xs, Ys, Zs = gen_nf_test_grid(experiment.cross_sectional_dimensions, [-experiment.voxel_spacing/2,experiment.voxel_spacing], experiment.voxel_spacing)
        to_use = np.arange(len(test_crds_full))
        test_coordinates = test_crds_full[to_use, :]

    # Check if we are iterating this variable
    if iterations > 1:
        # Initialize
        count = 0
        confidence_to_plot = np.zeros(iterations)
        parameter_space = np.linspace(start,stop,iterations)
        # Tell the user what we are doing
        print(f'Scanning over {parameter_name} from {start} to {stop} with {iterations} steps of {parameter_space[1]-parameter_space[0]}')
        # Loop over the parameter space
        for val in parameter_space:
            # Change experiment
            print(f'Testing {parameter_name} at: {val}')
            if parameter_number > 2 and parameter_number < 6: 
                # A translation - update the working_experiment
                working_experiment.detector_params[parameter_number] = val
                working_experiment.tVec_d[parameter_number-3] = val
            elif parameter_number <= 2:
                # A tilt - update the working_experiment
                # For user ease, I will have the input parameters in degrees about each axis
                # Some detector tilt information
                # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map
                # Grab original rotations
                xyzp_tilts_deg = np.multiply(rotations.angles_from_rmat_xyz(experiment.rMat_d),180.0/np.pi) # Passive (extrinsic) tilts XYZ
                # Reset the current value of the desired tilt
                xyzp_tilts_deg[parameter_number] = val
                # Define new rMat_d
                rMat_d = xfcapi.makeDetectorRotMat(np.multiply(xyzp_tilts_deg,np.pi/180.0))
                # Update the working_experiment
                working_experiment.rMat_d = rMat_d
                working_experiment.detector_params[0:3] = np.multiply(xyzp_tilts_deg,np.pi/180.0)
            else:
                # Chi angle
                working_experiment.chi = val*np.pi/180.0
                working_experiment.detector_params[6] = val*np.pi/180.0

            # Precompute orientaiton information (should need this for all, but it effects only chi?)
            precomputed_orientation_data = precompute_diffraction_data(working_experiment,controller,experiment.exp_maps)
            # Run the test
            raw_exp_maps, raw_confidence, raw_idx = test_orientations_at_coordinates(working_experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
            grain_map, confidence_map = process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=None,id_remap=experiment.remap)
            
            # Pull the sum of the confidence map
            confidence_to_plot[count] = np.sum(confidence_map)
            count = count + 1
        
        # Where was the center found?  
        # Weighted sum - does not work great
        #a = z_space; b = z_conf_to_plot - np.min(z_conf_to_plot); b = b/np.sum(b); working_z = np.sum(np.multiply(a,b)) 
        # Take the max - It's simple but will not throw any fits if we do not have a nice curve like a fitter might
        best_val = parameter_space[np.where(confidence_to_plot == np.max(confidence_to_plot))[0]]
        
        # Place the value where it needs to be
        if parameter_number > 2 and parameter_number < 6: 
            # A translation - update the working_experiment
            experiment.detector_params[parameter_number] = best_val
            experiment.tVec_d[parameter_number-3] = best_val
        elif parameter_number <= 2:
            # Tilt
            # For user ease, I will have the input parameters in degrees about each axis
            # Some detector tilt information
            # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map
            # Grab original rotations
            xyzp_tilts_deg = np.multiply(rotations.angles_from_rmat_xyz(experiment.rMat_d),180.0/np.pi) # Passive (extrinsic) tilts XYZ
            # Reset the current value of the desired tilt
            xyzp_tilts_deg[parameter_number] = best_val
            # Define new rMat_d
            rMat_d = xfcapi.makeDetectorRotMat(np.multiply(xyzp_tilts_deg,np.pi/180.0))
            # Update the working_experiment
            experiment.rMat_d = rMat_d
            experiment.detector_params[0:3] = np.multiply(xyzp_tilts_deg,np.pi/180.0)
        else:
            # Chi angle
            experiment.chi = val*np.pi/180.0
            experiment.detector_params[6] = val*np.pi/180.0
        
        # Plot the detector distance curve
        plt.figure()
        plt.plot(parameter_space,confidence_to_plot)
        plt.plot([best_val,best_val],[np.min(confidence_to_plot),np.max(confidence_to_plot)])
        plt.title(f'{parameter_name} Confidence Curve')
        plt.show(block=False)

        # Precompute orientaiton information (should need this for all, but it effects only chi?)
        precomputed_orientation_data = precompute_diffraction_data(experiment,controller,experiment.exp_maps)
        # Run the test
        raw_exp_maps, raw_confidence, raw_idx = test_orientations_at_coordinates(experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
        grain_map, confidence_map = process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=None,id_remap=experiment.remap)

        # Plot the new confidence map
        plt.figure()
        if parameter_number == 4:
            plt.imshow(confidence_map[:,:,0],clim=[0,1])
        else:
            plt.imshow(confidence_map[0,:,:],clim=[0,1])
        plt.title(f'Confidence Map with {parameter_name} = {best_val}')
        plt.show(block=False)

        # Quick update
        print(f'{parameter_name} found to produce highest confidence at {best_val}.\n\
              Scanning done.  The experiment has been updated with new value.\n\
              Update detector file if desired.')
        yaml_vals = experiment.detector_params[0:7]
        yaml_vals[0:3] = rotations.expMapOfQuat(rotations.quatOfRotMat(experiment.rMat_d))
        print(f'The updated values for the .ymal are:\n\
                  transform:\n\
                    translation:\n\
                    - {yaml_vals[3]}\n\
                    - {yaml_vals[4]}\n\
                    - {yaml_vals[5]}\n\
                    tilt:\n\
                    - {yaml_vals[0]}\n\
                    - {yaml_vals[1]}\n\
                    - {yaml_vals[2]}\n\
                    chi:{yaml_vals[6]}')
        return experiment
    elif iterations == 1:
        # Initialize
        val = start
        # Change experiment
        print(f'Testing {parameter_name} at: {val}')
        if parameter_number > 2 and parameter_number < 6: 
            # A translation - update the working_experiment
            working_experiment.detector_params[parameter_number] = val
            working_experiment.tVec_d[parameter_number-3] = val
        elif parameter_number <= 2:
            # A tilt - update the working_experiment
            # For user ease, I will have the input parameters in degrees about each axis
            # Some detector tilt information
            # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map
            # Grab original rotations
            xyzp_tilts_deg = np.multiply(rotations.angles_from_rmat_xyz(experiment.rMat_d),180.0/np.pi) # Passive (extrinsic) tilts XYZ
            # Reset the current value of the desired tilt
            xyzp_tilts_deg[parameter_number] = val
            # Define new rMat_d
            rMat_d = xfcapi.makeDetectorRotMat(np.multiply(xyzp_tilts_deg,np.pi/180.0))
            # Update the working_experiment
            working_experiment.rMat_d = rMat_d
            working_experiment.detector_params[0:3] = np.multiply(xyzp_tilts_deg,np.pi/180.0)
        else:
            # Chi angle
            working_experiment.chi = val*np.pi/180.0
            working_experiment.detector_params[6] = val*np.pi/180.0

        # Precompute orientaiton information (should need this for all, but it effects only chi?)
        precomputed_orientation_data = precompute_diffraction_data(working_experiment,controller,experiment.exp_maps)
        # Run the test
        raw_exp_maps, raw_confidence, raw_idx = test_orientations_at_coordinates(working_experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
        grain_map, confidence_map = process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=None,id_remap=experiment.remap)
        
        # Pull the sum of the confidence map
        confidence_to_plot = np.sum(confidence_map)
        # Plot the new confidence map
        plt.figure()
        if parameter_number == 4:
            plt.imshow(confidence_map[:,:,0],clim=[0,1])
        else:
            plt.imshow(confidence_map[0,:,:],clim=[0,1])
        plt.title(f'Confidence Map with {parameter_name} = {val}')
        plt.show(block=False)

        # Quick update
        print(f'{parameter_name} tested at {val}.\n\
              The experiment has been updated with this value.\n\
              Update detector file if desired.')
        yaml_vals = working_experiment.detector_params[0:7]
        yaml_vals[0:3] = rotations.expMapOfQuat(rotations.quatOfRotMat(working_experiment.rMat_d))
        print(f'The parameter values tested were:\n\
                  transform:\n\
                    translation:\n\
                    - {yaml_vals[3]}\n\
                    - {yaml_vals[4]}\n\
                    - {yaml_vals[5]}\n\
                    tilt:\n\
                    - {yaml_vals[0]}\n\
                    - {yaml_vals[1]}\n\
                    - {yaml_vals[2]}\n\
                    chi:{yaml_vals[6]}')
        return working_experiment
    else:
        print('Not iterating over any variable; testing current experiment.')
        yaml_vals = experiment.detector_params[0:7]
        yaml_vals[0:3] = rotations.expMapOfQuat(rotations.quatOfRotMat(experiment.rMat_d))
        print(f'The current values from the .ymal are:\n\
                  transform:\n\
                    translation:\n\
                    - {yaml_vals[3]}\n\
                    - {yaml_vals[4]}\n\
                    - {yaml_vals[5]}\n\
                    tilt:\n\
                    - {yaml_vals[0]}\n\
                    - {yaml_vals[1]}\n\
                    - {yaml_vals[2]}\n\
                    chi:{yaml_vals[6]}')
        # Precompute orientaiton information (should need this for all, but it effects only chi?)
        precomputed_orientation_data = precompute_diffraction_data(experiment,controller,experiment.exp_maps)
        # Run the test
        raw_exp_maps, raw_confidence, raw_idx = test_orientations_at_coordinates(experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
        grain_map, confidence_map = process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=None,id_remap=experiment.remap)
        # Plot
        plt.figure()
        plt.imshow(confidence_map[0,:,:],clim=[0,1])
        plt.title(f'Confidence Map')
        plt.show(block=False)
        return experiment


# %% ============================================================================
# RAW DATA PROCESSOR
# ===============================================================================
# Raw data processor
def process_raw_data(raw_confidence,raw_idx,volume_dims,mask=None,id_remap=None,raw_misorientation=None):
    # Assign the confidence to the correct voxel
    confidence_map = np.zeros(volume_dims)
    if mask is None:
        mask = np.ones(volume_dims,bool)
    confidence_map[mask] = raw_confidence
    
    # Apply remap if there is one
    if id_remap is not None:
        mapped_idx = id_remap[raw_idx]
    else:
        mapped_idx = raw_idx
    
    # Assign the indexing to the correct voxel
    grain_map = np.zeros(volume_dims)
    grain_map[mask] = mapped_idx

    # Reshape the misorientation if provided
    if raw_misorientation is not None:
        misorientation_map = np.zeros(volume_dims)
        misorientation_map[mask] = raw_misorientation
        return grain_map, confidence_map, misorientation_map
    else:
        return grain_map, confidence_map


# %% ============================================================================
# DIFFRACTION VOLUME STITCHERS
# ===============================================================================
# Stich individual diffraction volumes
def stitch_nf_diffraction_volumes(output_dir,output_stem,filepaths,material, 
                                  offsets,voxel_size,overlap=0,use_mask=0,ori_tol=0.0,remove_small_grains_under=0, 
                                  average_orientation=0, save_npz=0,save_h5=0,save_grains_out=0,suppress_plots=0,
                                  single_or_multiple_grains_out_files=0):
    """
        Goal: 
            
        Input:
            
        Output:

    """

    print('Loading grain map data.')
    # Some data lists initialization
    exp_map_list = []
    grain_map_list = []
    conf_map_list = []
    Xs_list = []
    Ys_list = []
    Zs_list = []
    nf_to_ff_id_map = []
    if use_mask == 1:
        masks = []
    
    # Load data into lists
    for i, p in enumerate(filepaths):
        nf_recon = np.load(p)
        grain_map_list.append(np.flip(nf_recon['grain_map'],0))
        conf_map_list.append(np.flip(nf_recon['confidence_map'],0))
        exp_map_list.append(nf_recon['ori_list'])
        Xs_list.append(np.flip(nf_recon['Xs'],0))
        Ys_list.append(np.flip(nf_recon['Ys'],0) - offsets[i])
        Zs_list.append(np.flip(nf_recon['Zs'],0))
        nf_to_ff_id_map.append(nf_recon['id_remap'])
        if use_mask == 1:
            masks.append(np.flip(nf_recon['tomo_mask'],0))

    # What are the dimensions of our arrays?
    dims = np.shape(grain_map_list[0]) # Since numpy dimensions are strange this is Y X Z

    # Initialize the merged data arrays
    num_vols = i + 1
    grain_map_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int32)
    confidence_map_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Xs_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Ys_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Zs_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    vertical_position_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    diffraction_volume = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int16)
    if use_mask == 1:
        mask_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int8)

    print('Grain map data Loaded.')
    print('Merging diffraction volumes.')

    # Run the merge
    if overlap != 0:
        # There is an overlap, use confidence to define which voxel to pull at each overlap
        # First assume there is no overlap
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop,:,:] = grain_map_list[vol][overlap:dims[0]-overlap,:,:]
            confidence_map_full[start:stop,:,:] = conf_map_list[vol][overlap:dims[0]-overlap,:,:]
            Xs_full[start:stop,:,:] = Xs_list[vol][overlap:dims[0]-overlap,:,:]
            Ys_full[start:stop,:,:] = Ys_list[vol][overlap:dims[0]-overlap,:,:]
            Zs_full[start:stop,:,:] = Zs_list[vol][overlap:dims[0]-overlap,:,:]
            vertical_position_full[start:stop,:,:] = offsets[vol]
            diffraction_volume[start:stop,:,:] = vol
            if use_mask == 1:
                mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
        
        # Now handle the overlap regions
        # Where are they?  They are the overlap voxels on either side of the volume division,
        # where the division-overlap region has to be checked against the first overlap voxels
        # of the next region, and a similar number of voxels into that region need to be checked
        # against the final voxels of the first
        # We have number of diffraction volumes - 1 overlap chunks
        division_line = dims[0] - overlap*2 # This is actually the first index of the next diffraction volume
        for overlap_region in np.arange(num_vols-1):
            # Handle one side
            # This will be true at voxels which need replacing
            replace_voxels = np.signbit(confidence_map_full[division_line-overlap:division_line,:,:]-conf_map_list[overlap_region+1][0:overlap,:,:])
            grain_map_full[division_line-overlap:division_line,:,:][replace_voxels] = grain_map_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            confidence_map_full[division_line-overlap:division_line,:,:][replace_voxels] = conf_map_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Xs_full[division_line-overlap:division_line,:,:][replace_voxels] = Xs_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Ys_full[division_line-overlap:division_line,:,:][replace_voxels] = Ys_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Zs_full[division_line-overlap:division_line,:,:][replace_voxels] = Zs_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            vertical_position_full[division_line-overlap:division_line,:,:][replace_voxels] = offsets[overlap_region+1]
            diffraction_volume[division_line-overlap:division_line,:,:][replace_voxels] = diffraction_volume[overlap_region+1][0:overlap,:,:][replace_voxels]
            if use_mask == 1:
                mask_full[division_line-overlap:division_line,:,:][replace_voxels] = masks[overlap_region+1][0:overlap,:,:][replace_voxels]
            division_line = division_line + dims[0] - overlap*2

    else:
        # There is no overlap, scans will be stacked directly
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop,:,:] = grain_map_list[vol][overlap:dims[0]-overlap,:,:]
            confidence_map_full[start:stop,:,:] = conf_map_list[vol][overlap:dims[0]-overlap,:,:]
            Xs_full[start:stop,:,:] = Xs_list[vol][overlap:dims[0]-overlap,:,:]
            Ys_full[start:stop,:,:] = Ys_list[vol][overlap:dims[0]-overlap,:,:]
            Zs_full[start:stop,:,:] = Zs_list[vol][overlap:dims[0]-overlap,:,:]
            vertical_position_full[start:stop,:,:] = offsets[vol]
            diffraction_volume[start:stop,:,:] = [vol]
            if use_mask == 1:
                mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2

    print('Diffraction volumes merged.')
    if ori_tol > 0.0:
        print('Voxelating orientation data.')
        # Voxelate the data
        dims = np.shape(grain_map_full)
        numel = np.prod(dims)
        id_array = np.arange(numel).reshape(dims).astype(int)
        ori_array = np.zeros([dims[0],dims[1],dims[2],3])
        for y in np.arange(dims[0]):
            for x in np.arange(dims[1]):
                for z in np.arange(dims[2]):
                    id = grain_map_full[y,x,z]
                    if id != -1:
                        which_scan = np.where(offsets.astype(np.float32) == vertical_position_full[y,x,z])[0][0]
                        idx = np.where(nf_to_ff_id_map[which_scan] == id)
                        ori_array[y,x,z,:] = exp_map_list[which_scan][idx]
        id_array[mask_full == 0] = -1
        print('Data voxeleated.')
        print('Merging grains.')

        # Grain merge methodology: calculate the misorientation of all voxels in volume and threshold.  
            # Detect only the blob in which the current voxel is located - call that the grain.  
            # Remove those grain's voxels from the test area and move on.  Do until we have nothing left.  
        grain_identification_complete = np.zeros(dims,dtype=bool)
        grain_map_merged = np.zeros(dims,dtype=np.int32)
        grain_map_merged[:] = -2 # Define everything to -2
        ori_map_merged = np.zeros([dims[0],dims[1],dims[2],3],dtype=np.float32)
        if use_mask == 1:
            grain_identification_complete[mask_full == 0] = True
            grain_map_merged[mask_full == 0] = -1
        grain = 0 # Starting grain id
        for y in np.arange(dims[0]):
            for x in np.arange(dims[1]):
                for z in np.arange(dims[2]):
                    if grain_identification_complete[y,x,z] == False: # Should not need to check mask...
                        ori = ori_array[y,x,z,:]
                        test_ids = id_array[~grain_identification_complete]
                        test_orientations = ori_array[~grain_identification_complete]

                        # Check misorientation
                        grain_quats = np.atleast_2d(rotations.quatOfExpMap(ori)).T
                        test_quats = np.atleast_2d(rotations.quatOfExpMap(test_orientations.T))
                        if np.shape(test_quats)[0] == 1:
                            [misorientations, a] = rotations.misorientation(grain_quats,test_quats.T)
                        else:
                            [misorientations, a] = rotations.misorientation(grain_quats,test_quats)
                        idx_to_merge = misorientations < np.radians(ori_tol)
                        ids_to_merge = test_ids[idx_to_merge]
                        voxels_to_merge = np.isin(id_array,ids_to_merge)

                        # Check for connectivity
                        labeled_array, num_features = scipy.ndimage.label(voxels_to_merge,structure=np.ones([3,3,3]))
                        if num_features > 1:
                            print('This grain has more at least one non-connected region.')
                            print('Only the region where the current voxel of interest resides will be labeled as this grain.')
                            where = labeled_array[y,x,z]
                            voxels_to_merge = labeled_array == where
                        else:
                            voxels_to_merge = labeled_array.astype('bool')

                        # Average the orientation
                        if average_orientation == 1:
                            # This has not been implemented since we currently only work with grain averaged
                            # orientations.  There may be a slight difference between nf scans but it *should*
                            # be negligible for the model HEDM materials we work with.  
                            print('Orientation averaging not implemented.  Useing single orientation.')
                            print('Implement orientation averaging if you need it.')
                            new_ori = ori
                        else:
                            new_ori = ori
                        
                        # We should not geneate any overlap but let's double check
                        if np.sum(grain_identification_complete[voxels_to_merge]) > 0:
                            print('We are trying to put a grain where one already exists...')

                        # Reasign stuff
                        grain_map_merged[voxels_to_merge] = grain
                        ori_map_merged[voxels_to_merge] = new_ori
                        grain_identification_complete[voxels_to_merge] = True

                        # Cleanup
                        print('Grain ' + str(grain) + ' identified.')
                        grain = grain + 1

        # A check
        if np.sum(np.isin(grain_map_merged,-2)) != 0:
            print('Looks like not every voxel was merged...')
        
        num_grains = grain # That +1 above handles grain 0
        new_ids = np.arange(num_grains)
        new_sizes = np.zeros(num_grains)
        new_oris = np.zeros([num_grains,3])
        for grain in new_ids:
            new_sizes[grain] = np.sum(grain_map_merged == grain)
            new_oris[grain,:] = ori_map_merged[grain_map_merged == grain][0]

        print('Grains merged.')

        print('Reassigning grain ids.')

        # Reorder the grains IDs such that the largest grain is grain 0
        final_ids = np.zeros(num_grains)
        final_sizes = np.zeros(num_grains)
        final_orientations = np.zeros([num_grains,3])
        idx = np.argsort(-new_sizes)
        final_grain_map = np.zeros(dims,np.int32)
        final_grain_map[:] = -2 # Define everything to -2
        if use_mask == 1:
            final_grain_map[mask_full == 0] = -1
        for grain in new_ids:
            final_grain_map[grain_map_merged == new_ids[idx[grain]]] = grain
            final_ids[grain] = grain
            final_orientations[grain,:] = new_oris[idx[grain]]
            final_sizes[grain] = new_sizes[idx[grain]]

        # A check
        if np.sum(np.isin(final_grain_map,-2)) != 0:
            print('Looks like not every voxel was merged...')

        print('Grain ids reassigned.')

        # Remove small grains if remove_small_grains_under is greater than zero
        if remove_small_grains_under > 0:
            print('Removeing grains smaller than ' + str(remove_small_grains_under) + ' voxels')
            print('Note that any voxels removed will only have the grain ID changed, the confidence value will not be touched')
            if use_mask == 1:
                grain_idx_to_keep = final_sizes>=remove_small_grains_under
                working_ids = final_ids[grain_idx_to_keep].astype(int)
                working_oris = final_orientations[grain_idx_to_keep]
                ids_to_remove = final_ids[~grain_idx_to_keep]
                working_grain_map = np.copy(final_grain_map)
                working_grain_map[working_grain_map>=ids_to_remove[0]] = -2
                for y in np.arange(dims[0]):
                    for x in np.arange(dims[1]):
                        for z in np.arange(dims[2]):
                            if working_grain_map[y,x,z] == -2:
                                mask = np.zeros(np.shape(working_grain_map))
                                mask[y,x,z] = 1
                                mask = scipy.ndimage.binary_dilation(mask,structure=np.ones([3,3,3]))
                                mask[mask_full == 0] = 0
                                mask[y,x,z] = 0
                                m,c = scipy.stats.mode(working_grain_map[mask], axis=None, keepdims=False)
                                working_grain_map[y,x,z] = m
                
                print('Done removing grains smaller than ' + str(remove_small_grains_under) + ' voxels')

                # Quick double check
                if np.sum(working_ids-np.unique(working_grain_map)[1:]) != 0:
                    print('Something went wrong with the small grain removal.')

                # Recheck sizes
                num_grains = np.shape(working_ids) # That +1 above handles grain 0
                working_sizes = np.zeros(num_grains)
                for grain in working_ids:
                    working_sizes[grain] = np.sum(working_grain_map == grain)

                print('Reassigning grain ids a final time.')

                # Reorder the grains IDs such that the largest grain is grain 0
                final_ids = np.zeros(num_grains,int)
                final_sizes = np.zeros(num_grains)
                final_orientations = np.zeros([num_grains[0],3])
                idx = np.argsort(-working_sizes)
                final_grain_map = np.zeros(dims,np.int32)
                if use_mask == 1:
                    final_grain_map[mask_full == 0] = -1
                for grain in working_ids:
                    final_grain_map[working_grain_map == working_ids[idx[grain]]] = grain
                    final_ids[grain] = grain
                    final_orientations[grain,:] = working_oris[idx[grain]]
                    final_sizes[grain] = working_sizes[idx[grain]]

                print('Grain ids reassigned.')
            else:
                print('You are not using a mask.  A mask is needed to define sample boundaries and correctly judge grain size.')

        if not suppress_plots:
            # Plot histogram of grain sizes
            fig, axs = plt.subplots(2,2,constrained_layout=True)
            fig.suptitle('Grain Size Statistics')
            # Plot number of voxels in all grains
            axs[0,0].hist(final_sizes,25)
            axs[0,0].title.set_text('Histogram of Grain \nSizes in Voxels')
            axs[0,0].set_xlabel('Number of Voxels')
            axs[0,0].set_ylabel('Frequency')
            # Plot number of voxels of just the small grains
            axs[0,1].hist(final_sizes[final_sizes<10],25)
            axs[0,1].title.set_text('Histogram of Grain \nSizes in Voxels (smaller grains)')
            axs[0,1].set_xlabel('Number of Voxels')
            axs[0,1].set_ylabel('Frequency')
            # Plot equivalent grain diameters for all grains
            axs[1,0].hist(np.multiply(final_sizes,6/math.pi*(voxel_size*1000)**3)**(1/3),25)
            axs[1,0].title.set_text('Histogram of Equivalent \nGrain Diameters (smaller grains)')
            axs[1,0].set_xlabel('Equivalent Grain Diameter (microns)')
            axs[1,0].set_ylabel('Frequency')
            # Plot equivalent grain diameters for the small grains
            axs[1,1].hist(np.multiply(final_sizes[final_sizes<10],6/math.pi*(voxel_size*1000)**3)**(1/3),25)
            axs[1,1].title.set_text('Histogram of Equivalent \nGrain Diameters (smaller grains)')
            axs[1,1].set_xlabel('Equivalent Grain Diameter (microns)')
            axs[1,1].set_ylabel('Frequency')
            # Wrap up
            plt.tight_layout()
            plt.show()

    else:
        print('Not merging or removing small grains.')
        print('Grain IDs will be left identical to the original diffraction volumes.')
        final_grain_map = grain_map_full
        final_orientations = np.reshape(exp_map_list,(np.shape(exp_map_list)[0]*np.shape(exp_map_list)[1],np.shape(exp_map_list)[2]))

    # Currently everything is upside down
    final_grain_map = np.flip(final_grain_map,0)
    confidence_map_full = np.flip(confidence_map_full,0)
    Xs_full = np.flip(Xs_full,0)
    Ys_full = np.flip(Ys_full,0)
    Zs_full = np.flip(Zs_full,0)
    diffraction_volume = np.flip(diffraction_volume,0)
    vertical_position_full = np.flip(vertical_position_full,0)
    if use_mask == 1:
        mask_full = np.flip(mask_full,0)

    # One final quick check
    if np.sum(final_ids-np.unique(final_grain_map)[1:]) != 0:
        print('Something went IDing the grains.')

    # Save stuff
    if save_h5 == 1:
        print('Writing h5 data...')
        if use_mask == 0:
            save_nf_data_for_paraview(output_dir,output_stem,final_grain_map,confidence_map_full,
                                            Xs_full,Ys_full,Zs_full,final_orientations,
                                            material,tomo_mask=None,id_remap=None,
                                            diffraction_volume_number=diffraction_volume)
        else:
            save_nf_data_for_paraview(output_dir,output_stem,final_grain_map,confidence_map_full,
                                Xs_full,Ys_full,Zs_full,final_orientations,
                                material,tomo_mask=mask_full,id_remap=None,
                                diffraction_volume_number=diffraction_volume)
    if save_npz == 1:
        print('Writing NPZ data...')
        if use_mask != 0:
            np.savez(output_dir+output_stem+'_merged_grain_map_data.npz',grain_map=final_grain_map,
                     confidence_map=confidence_map_full,Xs=Xs_full,Ys=Ys_full,Zs=Zs_full,
                     ori_list=final_orientations,id_remap=np.unique(final_grain_map),tomo_mask=mask_full,
                     diffraction_volume_number=diffraction_volume,vertical_position_full=vertical_position_full)
        else:
            np.savez(output_dir+output_stem+'_merged_grain_map_data.npz',grain_map=final_grain_map,
                     confidence_map=confidence_map_full,Xs=Xs_full,Ys=Ys_full,Zs=Zs_full,
                     ori_list=final_orientations,id_remap=np.unique(final_grain_map),
                     diffraction_volume_number=diffraction_volume,vertical_position_full=vertical_position_full,
                     tomo_mask=None,diffrction_volume_number=diffraction_volume)
    
    if save_grains_out == 1:
        # Find centroids for each grain with respect to the whole volume and each individual layer
        print('Finding centroids for the grains.out.')
        whole_volume_centroids = np.zeros([len(final_ids),3])
        diffraction_volume_centroids = np.zeros([num_vols,len(final_ids),3])
        for grain in final_ids:
            # Where is it in the whole volume?
            x_pos = np.mean(Xs_full[final_grain_map==grain])
            y_pos = np.mean(Ys_full[final_grain_map==grain])
            z_pos = np.mean(Zs_full[final_grain_map==grain])
            # Record it
            whole_volume_centroids[grain,:] = [x_pos,y_pos,z_pos]
            # Run through each diffraction volume and grab the centroid in each
            for vol in np.arange(num_vols):
                # Create diffraction_vol mask on the grain map
                mask = np.copy(final_grain_map)
                mask[diffraction_volume != vol] = -1
                # Is it in this vol?
                if np.sum(mask==grain) > 0:
                    x_pos = np.mean(Xs_full[mask==grain])
                    y_pos = np.mean(Ys_full[mask==grain])
                    z_pos = np.mean(Zs_full[mask==grain])
                    # Record it
                    diffraction_volume_centroids[vol,grain,:] = [x_pos,y_pos,z_pos]

        print('Writing grains.out data...')
        if single_or_multiple_grains_out_files == 0:
            print('Writing single grains.out with centroids for whole volume...')
            gw = instrument.GrainDataWriter(
                os.path.join(output_dir, output_stem) + '_whole_volume_grains.out'
            )
            for gid, ori in enumerate(final_orientations):
                grain_params = np.hstack([ori, whole_volume_centroids[gid,:], constants.identity_6x1])
                gw.dump_grain(gid, 1., 0., grain_params)
            gw.close()
        elif single_or_multiple_grains_out_files == 1:
            print('Writing multiple grains.out with centroids for each volume...')
            for vol in np.arange(num_vols):
                gw = instrument.GrainDataWriter(
                    os.path.join(output_dir, output_stem) + '_diff_vol_num_' + str(vol) + '_grains.out'
                )
                for gid, ori in enumerate(final_orientations):
                    grain_params = np.hstack([ori, diffraction_volume_centroids[vol,gid,:], constants.identity_6x1])
                    gw.dump_grain(gid, 1., 0., grain_params)
                gw.close()

    return final_orientations

# %% ============================================================================
# MISSING GRAINS FUNCTIONS
# ===============================================================================
# Find low confidence regions within the diffraction volume
def generate_low_confidence_test_coordinates(starting_reconstruction,confidence_threshold=0.5,how_sparse=0,errode_free_surface=1):
    # Data path must point to a npz save of either a merged volume or single volume
    # Within this file there MUST be:
        # confidence, X, Y, Z, and mask
        # If you do not have a mask, make one.  

    # Load the data
    confidence_map = starting_reconstruction['confidence_map']
    mask = starting_reconstruction['tomo_mask']
    Xs = starting_reconstruction['Xs']
    Ys = starting_reconstruction['Ys']
    Zs = starting_reconstruction['Zs']

    # Create a id array
    dims = np.shape(confidence_map)
    id_array = np.zeros(dims)
    id = 1
    for y in np.arange(dims[0]):
        for x in np.arange(dims[1]):
            for z in np.arange(dims[2]):
                id_array[y,x,z] = id
                id = id + 1

    # Remove the free surface
    if errode_free_surface == 1:
        # Errode all 2D blobs - will get free surface and interior blobs
        temp = np.copy(mask)
        for i in np.arange(np.shape(mask)[0]):
            temp[i,:,:] = scipy.ndimage.binary_erosion(mask[i,:,:])
        mask = temp

    # Find the regions of low confidence
    low_confidence_map = confidence_map<confidence_threshold
    low_confidence_map[mask == 0] = 0

    # Create sparse grid mask
    # Make all even indices 1 and odd indices 0
    sparse_mask = np.ones(np.shape(low_confidence_map))
    if how_sparse == 0:
        # Not sparce at all
        print('Returning all voxels under the confidence threshold')
    elif how_sparse == 1:
        # Some sparcing applied
        sparse_mask[::2,:,:] = 0
        sparse_mask[:,::2,:] = 0
        sparse_mask[:,:,::2] = 0
        sparse_mask = scipy.ndimage.binary_dilation(sparse_mask,structure=np.array([[[1,0,1],[0, 0, 0],[1,0,1]],
                                                                    [[0,0,0],[0, 1, 0],[0,0,0]],
                                                                    [[1,0,1],[0, 0, 0],[1,0,1]]]))
        print('Returning most voxels under the confidence threshold')
    elif how_sparse == 2:
        # More sparcing applied
        sparse_mask[::2,:,:] = 0
        sparse_mask[:,::2,:] = 0
        sparse_mask[:,:,::2] = 0
        print('Returning some voxels under the confidence threshold')

    # Scatter shot low confidence map
    point_map = np.logical_and(low_confidence_map,sparse_mask)

    # What are these poistions in the lab coordinate system?
    Xs_positions = Xs[point_map]
    Ys_positions = Ys[point_map]
    Zs_positions = Zs[point_map]
    ids = id_array[point_map]
    test_coordiates = np.vstack([Xs_positions, Ys_positions, Zs_positions]).T

    print(f'{np.shape(test_coordiates)[0]} test coordinates found')
    return test_coordiates, ids

# Sample orientation space
def uniform_fundamental_zone_sampling(point_group_number,average_angular_spacing_in_deg=3.0):
    # Below is the list of all of the point groups, in order by thier Schoenfiles notation
    # The list starts with 1 so a point_group_number = 5 will return you c2h

    # SYM_GL_PG = {
    # 'c1': '1a',  # only identity rotation
    # 'ci': '1h',  # only inversion operation
    # 'c2': '1c',  # 2-fold rotation about z
    # 'cs': '1j',
    # 'c2h': '2ch',
    # 'd2': '2bc',
    # 'c2v': '2bj',
    # 'd2h': '3bch',
    # 'c4': '1g',
    # 's4': '1m',
    # 'c4h': '2gh',
    # 'd4': '2cg',
    # 'c4v': '2gj',
    # 'd2d': '2cm',
    # 'd4h': '3cgh',
    # 'c3': '1n',
    # 's6': '2hn',
    # 'd3': '2en',
    # 'c3v': '2kn ',
    # 'd3d': '3fhn',
    # 'c6': '2bn',
    # 'c3h': '2in',
    # 'c6h': '3bhn',
    # 'd6': '3ben',
    # 'c6v': '3bkn',
    # 'd3h': '3ikn',
    # 'd6h': '4benh',
    # 't': '2cd',
    # 'th': '3cdh',
    # 'o': '2dg',
    # 'td': '2dm',
    # 'oh': '3dgh'
    # }

    # Sample the fundamental zone
    s = sampleRFZ(point_group_number,average_angular_spacing=average_angular_spacing_in_deg)
    print(f"Sampled {np.shape(s.orientations)[0]} orientations within fundamental region of point group {point_group_number} with {average_angular_spacing_in_deg} degree step size")
    return s.orientations


# %% ============================================================================
# TEST GRID GENERATION FUNCTIONS
# ===============================================================================
def gen_nf_test_grid(cross_sectional_dim, v_bnds, voxel_spacing):

    Zs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)
    Xs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)


    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(Xs_list,v_bnds[0],Zs_list)
    else:
        Xs,Ys,Zs=np.meshgrid(Xs_list,np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),Zs_list)
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))



    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)


    return test_crds, n_crds, Xs, Ys, Zs

def gen_nf_test_grid_vertical(cross_sectional_dim, v_bnds, voxel_spacing):

    # This is designed only for calibration of a single slice along the z direction
    # So the Z thickness will be one voxel
    Zs_list=np.arange(-voxel_spacing/2.,voxel_spacing/2,voxel_spacing)
    Xs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)


    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(Xs_list,v_bnds[0],Zs_list)
    else:
        Xs,Ys,Zs=np.meshgrid(Xs_list,np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),Zs_list)
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))



    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)


    return test_crds, n_crds, Xs, Ys, Zs

def generate_test_coordinates(cross_sectional_dim, v_bnds, voxel_spacing,
                              mask_data_file=None,vertical_motor_position=0.0):
    if mask_data_file is not None:
        # Load the mask
        mask_data = np.load(mask_data_file)

        mask_full = mask_data['mask']
        Xs_mask = mask_data['Xs']
        Ys_mask = mask_data['Ys']+(vertical_motor_position)
        Zs_mask = mask_data['Zs']
        voxel_spacing = mask_data['voxel_spacing']

        # need to think about how to handle a single layer in this context
        tomo_layer_centers = np.squeeze(Ys_mask[:, 0, 0])
        above = np.where(tomo_layer_centers >= v_bnds[0])
        below = np.where(tomo_layer_centers < v_bnds[1])

        in_bnds = np.intersect1d(above, below)

        mask = mask_full[in_bnds]
        Xs = Xs_mask[in_bnds]
        Ys = Ys_mask[in_bnds]
        Zs = Zs_mask[in_bnds]

        test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
        
        to_use = np.squeeze(np.where(mask.flatten()))
    else:
        test_crds_full, n_crds, Xs, Ys, Zs = gen_nf_test_grid(
            cross_sectional_dim, v_bnds, voxel_spacing)
        to_use = np.arange(len(test_crds_full))
        mask = np.ones(Xs.shape,bool)

    test_coordinates = test_crds_full[to_use, :]
    return Xs, Ys, Zs, mask, test_coordinates

