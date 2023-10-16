NF-HEDM and Tomo for Masking README (written by seg246)

General Intro:
- how does this look as a workflow in the gui

    - NF-HEDM raw images are processed with raw_to_binary_nf_image_processor.py
    - Standard NF-HEDM reconstructions are completed with reconstruct_NF.py
    - calibrator
    - Tomo reconstructions are done in CHAP then processed with process_CHAP_tomo_for_NF_mask.py
    - Missing grains in individual NF-HEDM diffraction volumes are searched for with find_missing_grains.py
    - Separate NF reconstruction from multuple diffraction volumes are merged with stitch_multiple_diffraction_volumes.py

Order of Events:
    - Note that this is a suggested order of events and process.  If you have good reason, all of this process can 
        be changed.  Think critically about each step and if you are unsure why something is suggested please speak
        with your beamline scientist.  
    - If you don't have tomography to give NF-HEDM a mask then skip to step 5

    1. A detector.yml file is put together (manually) with values for tomo
        - This detector file is different than the detector file for NF and only has information about the 
            pixel size, magnification, and detector dimensions.  
        - Start with one of the provided templates.  
    2. A pipeline.yml file is put together (manually) with settings and file paths for tomo
        - Start with one of the provided templates.  
    3. CHAP tomo is run to output a Nexus file with the reconstructed tomography
        - From the CHESS compute farm in a linux terminal run:
            source /nfs/chess/sw/miniconda3_msnc/bin/activate
            conda activate CHAP_tomo
            CHAP pipeline.yml
        - This will output a Nexus file named as dictated in the pipeline.yml file
    4. Run process_CHAP_tomo_for_NF_mask.py with the Nexus file as an input
        - This needs to be run in a HEXRD environment
        - Since this code is a very visual code, run it in an IDE like vscode
        - This will output a tomography mask of a desired voxel size for NF
        - Choose your filepath and names carefully as they will be idential to those used in NF
    5. FF-HEDM is run (calibration,indexing, and fitting) to provide a list of test orientations for NF-HEDM (a grains.out file)
    6. NF-HEDM images are binarized with raw_to_binary_nf_image_processor.py
        - There are a number of fitting options, the guassian blur is often the simplest and most robust image processor
        - This will output both a binarized h5 stack of the images for NF as well as a h5 with image omega positions
        - Since this code is a very visual code, run it in an IDE like vscode
    7. A materials.h5 is written from HEXRDGUI to tell NF-HEDM which HKLs to use in the sample material.  
        - The HKLs to use will not nessesarially match FF-HEDM.  It is suggested to draw the NF-HEDM geometry to 
            determine which HKLs will intercept the detector - use the measured NF detector distance and refine 
            once NF calibration is complete.  
    8. A detector.yml file is made to provide detector details to NF-HEDM
        - Start with one of the provided templates.  
        - Note this is different than the one for tomo
        - All tilts and oscilation stage values should be set to zero
        - Your X translation is usually zero if the sample was placed in the center of the detector (check lab notes)
        - Your Y translation is usually zero if the detector was centered about the incoming beam (check lab notes)
        - Your Z translation (sample to detector distance) should have been measured by a ruler during the experiment (check lab notes)
    9. NF-HEDM calibration is completed in reconstruct_nf.py
        - I suggest using v_bnds of [-0.0025, 0.0025] and a voxel size of 0.005
        - The calibrator first iterates over Z, then X, and finally Y.  Start with large bounds with a coarse step and refine. 
        - If your detector was centered about the incoming beam, leave Y at zero.  
        - The tilts are not touched, talk to your beamline scientist if these are something which should be worried about
        - Chi is not touched.  I suggest leaving this as zero if you have a gripped RAMS sample.  
        - Be sure to manually update your detector.yml file each time you run the iterator.  
    10. Reconstruct the full diffraction volume in reconstruct_nf.py
        - If you have multiple diffraction volumes you can reconstruct those as well.  
        - If you have a gripped RAMS sample, the NF calibration should not change between diffraction volumes.  
        - The detector Y position must not change between diffraction volumes if you intend to merge them.  
        - This script can output either an .npz or .h5 (with or without .xdmf for viewing in Paraview)
    11. Find missing grains with find_missing_grains.py
        - This is by far the most time consuming step and should only be done if you need to.  
        - This script reads in the reconstruction .npz from step 10
        - It will output an updated grains.out and updated reconstruction (if desired)
        - Note that a mask is required for this script to function.  
    12. Merge the final diffraction volumes with stitch_multiple_diffraction_volumes.py
        - This script will stitch together multiple reconstructed diffraction volumes and output a new grains.out
        - A merged h5 can also be written
        - This merged grains.out can be used to keep all your grain IDs together to help with book-keeping in FF
        - Just be careful about which grains should be in each diffraction volume.  