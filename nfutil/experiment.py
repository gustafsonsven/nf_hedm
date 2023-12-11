class Experiment():
    ...



# Generate the experiment
def generate_experiment(cfg):
    analysis_name = cfg.analysis_name # The name you want all your output files to have within thier filename (relevant to the sample)
    main_directory = cfg.main_directory
    output_directory = cfg.output_directory
    output_plot_check = cfg.output_plot_check

    # reconstruction size parameters
    # TODO: make cross sectional dims changeable
    cross_sectional_dim = cfg.reconstruction.cross_sectional_dimensions # [1.0,1.0] # mm - [perpendicualr, parallel] to incoming beam direction
    voxel_spacing = cfg.reconstruction.voxel_spacing
    beam_vertical_span = cfg.experiment.beam_vertical_span
    vertical_bounds = cfg.reconstruction.desired_vertical_span
    ncpus = cfg.multiprocessing.num_cpus
    chunk_size = cfg.multiprocessing.chunk_size
    #check = cfg.multiprocessing.check
    #limit = cfg.multiprocessing.limit
    #generate = cfg.multiprocessing.generate
    #max_RAM = cfg.multiprocessing.max_RAM  # this is in bytes
    
    # Load the grains.out data
    ff_data=np.loadtxt(cfg.input_files.grains_out_file)
    # Tell the user what we are doing so they know
    print(f'Grain data loaded from: {cfg.input_files.grains_out_file}')

    # Unpack grain data
    completeness = ff_data[:,1] # Completness
    chi2 = ff_data[:,2] # Chi^2
    exp_maps = ff_data[:,3:6] # Orientations
    t_vec_s = ff_data[:,6:9] # Grain centroid positions
    # How many grains do we have total?
    n_grains_pre_cut = exp_maps.shape[0]

    # Trim grain information so that we pull only the grains that we want
    comp_thresh = cfg.experiment.comp_thresh
    chi2_thresh = cfg.experiment.chi2_thresh
    cut = np.where(np.logical_and(completeness>comp_thresh,chi2<chi2_thresh))[0]
    exp_maps = exp_maps[cut,:] # Orientations
    t_vec_s = t_vec_s[cut,:] # Grain centroid positions

    # How many grains do we have after the cull?
    n_grains = exp_maps.shape[0]
    # Tell the user what we are doing so they know
    print(f'{n_grains} grains out of a total {n_grains_pre_cut} found to satisfy completness and chi^2 thresholds.')

    # Load the images
    images_filename = output_directory + os.sep + analysis_name + '_packaged_images.npy'
    if os.path.isfile(images_filename):
        # We have an image stack to load
        print(f'Images to be loaded from: {images_filename}')
        image_stack = np.load(images_filename)
        nframes = np.shape(image_stack)[0]
    # else:
        # TODO: Add old image load routine?
        # nframes = cfg.images.nframes

    # Load the omega edges
    omega_edges_filename = output_directory + os.sep + analysis_name + '_omega_edges_deg.npy'
    if os.path.isfile(omega_edges_filename):
        # Load the omega edges - first value is the starting ome position of first image's slew, last value is the end position of the final image's slew
        omega_edges_deg = np.load(omega_edges_filename)
    else:
        # Define omega edges manually
        omega_edges_deg = np.linspace(cfg.experiment.omega_start, cfg.experiment.omega_stop, num=nframes+1)

    # Shift in omega positive or negative by X number of images
    num_img_to_shift = cfg.experiment.shift_images_in_omega
    if num_img_to_shift > 0:
        # Moving positive omega so first image is not at zero, but further along
        # Using the mean omega step size - change if you need to
        omega_edges_deg = omega_edges_deg + num_img_to_shift*np.mean(np.gradient(omega_edges_deg))
    elif num_img_to_shift < 0:
        # For whatever reason the multiprocessor does not like negative numbers, trim the stack
        image_stack = image_stack[np.abs(num_img_to_shift):,:,:]
        nframes = np.shape(image_stack)[0]
        omega_edges_deg = omega_edges_deg[:num_img_to_shift]
    # Define omega edges in radians
    ome_edges = omega_edges_deg*np.pi/180


    # Define variables in degrees
    # Omega range is the experimental span of omega space
    ome_range_deg = [(omega_edges_deg[0],omega_edges_deg[nframes])]  # Degrees
    # Omega period is the range in which your omega space lies (often 0 to 360 or -180 to 180)
    ome_period_deg = (ome_range_deg[0][0], ome_range_deg[0][0]+360.) # Degrees
    # Define variables in radians
    ome_period = (ome_period_deg[0]*np.pi/180.,ome_period_deg[1]*np.pi/180.)
    ome_range = [(ome_range_deg[0][0]*np.pi/180.,ome_range_deg[0][1]*np.pi/180.)]

    # Load the detector data
    if os.path.isfile(cfg.input_files.detector_file):
        instr = load_instrument(cfg.input_files.detector_file)
        print(f'Detector data loaded from: {cfg.input_files.detector_file}')
    elif os.path.isfile(main_directory + os.sep + cfg.input_files.detector_file):
        instr = load_instrument(main_directory + os.sep + cfg.input_files.detector_file)
        print(f'Detector data loaded from: {main_directory + os.sep + cfg.input_files.detector_file}')
    else:
        print('No detector file found.')
    panel = next(iter(instr.detectors.values()))

    # Sample transformation parameters
    chi = instr.chi
    tVec_s = instr.tvec
    # Some detector tilt information
    # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map 
    rMat_d = panel.rmat # Generated by xfcapi.makeRotMatOfExpMap(tilt) where tilt are directly read in from the .yaml as a exp_map 
    tilt_angles_xyzp = np.asarray(rotations.angles_from_rmat_xyz(rMat_d)) # These are needed for xrdutil.simulateGVecs where they are converted to a rotation matrix via xfcapi.makeDetectorRotMat(detector_params[:3]) which reads in tiltAngles = [gamma_Xl, gamma_Yl, gamma_Zl] in radians
    tVec_d = panel.tvec

    # Pixel information
    row_ps = panel.pixel_size_row
    col_ps = panel.pixel_size_col
    pixel_size = (row_ps, col_ps)
    nrows = panel.rows
    ncols = panel.cols
    # Detector panel dimension information
    panel_dims = [tuple(panel.corner_ll),
                  tuple(panel.corner_ur)]
    x_col_edges = panel.col_edge_vec
    y_row_edges = panel.row_edge_vec
    # What is the max tth possible on the detector?
    max_pixel_tth = instrument.max_tth(instr)
    # Package detector parameters
    detector_params = np.hstack([tilt_angles_xyzp, tVec_d, chi, tVec_s])
    distortion = panel.distortion  # TODO: This is currently not used.

    # Parametrization for faster computation
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     ome_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       ome_edges[1] - ome_edges[0]])
    inv_deltas = 1.0/deltas
    clip_vals = np.array([ncols, nrows])

    # General crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", cfg.experiment.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    # Load the materials file
    if os.path.isfile(cfg.input_files.materials_file):
        mats = material.load_materials_hdf5(cfg.input_files.materials_file, dmin=dmin,kev=beam_energy)
        print(f'{cfg.experiment.material_name} material data loaded from: {cfg.input_files.materials_file}')
    elif os.path.isfile(main_directory + os.sep + cfg.input_files.materials_file):
        mats = material.load_materials_hdf5(main_directory + os.sep + cfg.input_files.materials_file, dmin=dmin,kev=beam_energy)
        print(f'{cfg.experiment.material_name} material data loaded from: {main_directory + os.sep + cfg.input_files.materials_file}')
    else:
        print('No materials file found.')


    pd = mats[cfg.experiment.material_name].planeData


    # Check and set the max tth desired or use the detector value
    max_tth = cfg.experiment.max_tth
    if max_tth is not None:
         pd.tThMax = np.amax(np.radians(max_tth))
    else:
        pd.tThMax = np.amax(max_pixel_tth)

    # Pull the beamstop
    if len(cfg.reconstruction.beam_stop) == 2:
        # Load from configuration
        beam_stop_parms = cfg.reconstruction.beam_stop
        # We need to make a mask out of the parameters
        beam_stop_mask = np.zeros([nrows,ncols],bool)
        # What is the middle position of the beamstop
        middle_idx = int(np.floor(nrows/2.) + np.round(beam_stop_parms[0]/col_ps))
        # How thick is the beamstop
        half_width = int(beam_stop_parms[1]/col_ps/2)
        # Make the beamstop all the way across the image
        beam_stop_mask[middle_idx - half_width:middle_idx + half_width,:] = 1
        # Set the mask
        beam_stop_parms = beam_stop_mask
    else:
        # Load
        try:
            beam_stop_parms = np.load(cfg.reconstruction.beam_stop)
            print(f'Loaded beam stop mask from: {cfg.reconstruction.beam_stop}')
        except:
            beam_stop_parms = np.load(output_directory + os.sep + analysis_name + '_beamstop_mask.npy')
            print(f'Loaded beam stop mask from: {output_directory + os.sep + analysis_name + "_beamstop_mask.npy"}')


    # Package up the experiment
    experiment = argparse.Namespace()
    # grains related information
    experiment.n_grains = n_grains
    experiment.exp_maps = exp_maps
    experiment.plane_data = pd
    experiment.detector_params = detector_params
    experiment.pixel_size = pixel_size
    experiment.ome_range = ome_range
    experiment.ome_period = ome_period
    experiment.x_col_edges = x_col_edges
    experiment.y_row_edges = y_row_edges
    experiment.ome_edges = ome_edges
    experiment.ncols = ncols
    experiment.nrows = nrows
    experiment.nframes = nframes  # used only in simulate...
    experiment.rMat_d = rMat_d
    experiment.tVec_d = tVec_d
    experiment.chi = chi  # note this is used to compute S... why is it needed?
    experiment.tVec_s = tVec_s
    experiment.distortion = distortion
    experiment.panel_dims = panel_dims  # used only in simulate...
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.clip_vals = clip_vals
    experiment.bsp = beam_stop_parms
    experiment.mat = mats
    experiment.material_name = cfg.experiment.material_name
    experiment.remap = cut
    experiment.vertical_bounds = vertical_bounds
    experiment.beam_vertical_span = beam_vertical_span
    experiment.cross_sectional_dimensions = cross_sectional_dim
    experiment.voxel_spacing = voxel_spacing
    experiment.ncpus = ncpus
    experiment.chunk_size = chunk_size
    experiment.analysis_name = analysis_name
    experiment.main_directory = main_directory
    experiment.output_directory = output_directory
    experiment.output_plot_check = output_plot_check
    experiment.point_group_number = cfg.experiment.point_group_number
    experiment.t_vec_s = t_vec_s
    experiment.centroid_serach_radius = cfg.reconstruction.centroid_serach_radius
    experiment.expand_radius_confidence_threshold = cfg.reconstruction.expand_radius_confidence_threshold

    # Tomo parameters
    if cfg.reconstruction.tomography is None:
        experiment.mask_filepath = None
        experiment.vertical_motor_position = None
    else:
        # TODO: Add project through single layer
        experiment.mask_filepath = cfg.reconstruction.tomography.mask_filepath
        experiment.vertical_motor_position = cfg.reconstruction.tomography.vertical_motor_position
    
    # Misorientation
    if cfg.reconstruction.misorientation:
        misorientation_bnd = cfg.reconstruction.misorientation['misorientation_bnd']
        misorientation_spacing = cfg.reconstruction.misorientation['misorientation_spacing']
        experiment.misorientation_bound_rad = misorientation_bnd*np.pi/180.
        experiment.misorientation_step_rad = misorientation_spacing*np.pi/180.
        experiment.refine_yes_no = 1
    # Tomo mask
    if cfg.reconstruction.tomography:
        experiment.mask_filepath = cfg.reconstruction.tomography['mask_filepath']
        experiment.vertical_motor_position = cfg.reconstruction.tomography['vertical_motor_position']
        experiment.use_single_layer = cfg.reconstruction.tomography['use_single_layer']
    
    if cfg.reconstruction.missing_grains:
        experiment.reconstructed_data_path = cfg.reconstruction.missing_grains['reconstructed_data_path']
        experiment.ori_grid_spacing = cfg.reconstruction.missing_grains['ori_grid_spacing']
        experiment.confidence_threshold = cfg.reconstruction.missing_grains['confidence_threshold']
        experiment.low_confidence_sparsing = cfg.reconstruction.missing_grains['low_confidence_sparsing']
        experiment.errode_free_surface = cfg.reconstruction.missing_grains['errode_free_surface']
        experiment.coord_cutoff_scale = cfg.reconstruction.missing_grains['coord_cutoff_scale']
        experiment.iter_cutoff = cfg.reconstruction.missing_grains['iter_cutoff']
        experiment.re_run_and_save = cfg.reconstruction.missing_grains['re_run_and_save']


    return experiment, image_stack

