import numpy as np
import matplotlib.pyplot as plt

from hexrd.transforms import xfcapi

# %% ============================================================================
# DATA PLOTTERS
# ===============================================================================
# Pulls IPF colors for each grain
def generate_ori_map(grain_map, exp_maps,mat,id_remap=None):
    # Init
    n_grains=len(exp_maps)
    if np.shape(grain_map)[0] == 1:
        grains_map_thin=np.squeeze(grain_map)
        rgb_image = np.zeros([grains_map_thin.shape[0], grains_map_thin.shape[1], 3], dtype='float32')
        # Colormapping
        for ii in np.arange(n_grains):
            if id_remap is not None:
                this_grain = np.where(np.squeeze(grains_map_thin) == id_remap[ii])
            else:
                this_grain = np.where(np.squeeze(grains_map_thin) == ii)
            if np.sum(this_grain) > 0:
                ori = exp_maps[ii, :]
                rmats = xfcapi.makeRotMatOfExpMap(ori)
                rgb = mat.unitcell.color_orientations(
                    rmats, ref_dir=np.array([0., 1., 0.]))
                rgb_image[this_grain[0], this_grain[1], 0] = rgb[0][0]
                rgb_image[this_grain[0], this_grain[1], 1] = rgb[0][1]
                rgb_image[this_grain[0], this_grain[1], 2] = rgb[0][2]
        # Redimension
        rgb_image = np.expand_dims(rgb_image,0)
    else:
        rgb_image = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 3], dtype='float32')
        # Colormapping
        for ii in np.arange(n_grains):
            if id_remap is not None:
                this_grain = np.where(np.squeeze(grain_map) == id_remap[ii])
            else:
                this_grain = np.where(np.squeeze(grain_map) == ii)
            if np.sum(this_grain) > 0:
                ori = exp_maps[ii, :]
                rmats = xfcapi.makeRotMatOfExpMap(ori)
                rgb = mat.unitcell.color_orientations(
                    rmats, ref_dir=np.array([0., 1., 0.]))
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 0] = rgb[0][0]
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 1] = rgb[0][1]
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 2] = rgb[0][2]


    return rgb_image

# An IPF and confidence map plotter
def plot_ori_map(grain_map, confidence_map, Xs, Zs, exp_maps, 
                 layer_no,mat,id_remap=None, conf_thresh=None):
    # Init
    grains_plot=np.squeeze(grain_map[layer_no,:,:])
    conf_plot=np.squeeze(confidence_map[layer_no,:,:])
    n_grains=len(exp_maps)
    rgb_image = np.zeros(
        [grains_plot.shape[0], grains_plot.shape[1], 3], dtype='float32')
    # Color mapping
    for ii in np.arange(n_grains):
        if id_remap is not None:
            this_grain = np.where(np.squeeze(grains_plot) == id_remap[ii])
        else:
            this_grain = np.where(np.squeeze(grains_plot) == ii)
        if np.sum(this_grain) > 0:
            ori = exp_maps[ii, :]
            rmats = xfcapi.makeRotMatOfExpMap(ori)
            rgb = mat.unitcell.color_orientations(
                rmats, ref_dir=np.array([0., 1., 0.]))
            
            rgb_image[this_grain[0], this_grain[1], 0] = rgb[0][0]
            rgb_image[this_grain[0], this_grain[1], 1] = rgb[0][1]
            rgb_image[this_grain[0], this_grain[1], 2] = rgb[0][2]
    # Define axes
    num_markers = 5
    x_axis = Xs[0,:,0] # This is the vertical axis
    no_axis = np.linspace(0,np.shape(x_axis)[0],num=num_markers)
    x_axis = np.linspace(x_axis[0],x_axis[-1],num=num_markers)
    z_axis = Zs[0,0,:] # This is the horizontal axis
    z_axis = np.linspace(z_axis[0],z_axis[-1],num=num_markers)
    # Plot
    if conf_thresh is not None:
        # Apply masking
        mask = conf_plot > conf_thresh
        rgb_image[:,:,0] = np.multiply(rgb_image[:,:,0],mask)
        rgb_image[:,:,1] = np.multiply(rgb_image[:,:,1],mask)
        rgb_image[:,:,2] = np.multiply(rgb_image[:,:,2],mask)
        conf_plot = np.multiply(conf_plot,mask)
        grains_plot = np.multiply(grains_plot,mask).astype(int)
        # Start Figure
        fig, axs = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle('Layer %d' % layer_no)
        # Plot IPF
        ax1 = axs[0,0].imshow(rgb_image,interpolation='none')
        axs[0,0].title.set_text('IPF')
        axs[0,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,0].set_xlabel('Z Position')
        axs[0,0].set_ylabel('X Position')
        # Plot Grain Map
        ax2 = axs[0,1].imshow(grains_plot,interpolation='none',cmap='hot')
        axs[0,1].title.set_text('Grain Map')
        axs[0,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,1].set_xlabel('Z Position')
        axs[0,1].set_ylabel('X Position')
        plt.colorbar(ax2)
        # Plot Confidence
        ax3 = axs[1,0].imshow(conf_plot,interpolation='none',cmap='bone')
        axs[1,0].title.set_text('Confidence')
        axs[1,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,0].set_xlabel('Z Position')
        axs[1,0].set_ylabel('X Position')
        plt.colorbar(ax3)
        # Plot Filler Plot
        ax4 = axs[1,1].imshow(np.zeros(np.shape(conf_plot)),interpolation='none')
        axs[1,1].title.set_text('Filler')
        axs[1,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,1].set_xlabel('Z Position')
        axs[1,1].set_ylabel('X Position')
        # Wrap up
        plt.show()
    else:
        # Start Figure
        fig, axs = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle('Layer %d' % layer_no)
        # Plot IPF
        ax1 = axs[0,0].imshow(rgb_image,interpolation='none')
        axs[0,0].title.set_text('IPF')
        axs[0,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,0].set_xlabel('Z Position')
        axs[0,0].set_ylabel('X Position')
        # Plot Grain Map
        ax2 = axs[0,1].imshow(grains_plot,interpolation='none',cmap='hot')
        axs[0,1].title.set_text('Grain Map')
        axs[0,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,1].set_xlabel('Z Position')
        axs[0,1].set_ylabel('X Position')
        plt.colorbar(ax2)
        # Plot Confidence
        ax3 = axs[1,0].imshow(conf_plot,interpolation='none',cmap='bone')
        axs[1,0].title.set_text('Confidence')
        axs[1,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,0].set_xlabel('Z Position')
        axs[1,0].set_ylabel('X Position')
        plt.colorbar(ax3)
        # Plot Filler Plot
        ax4 = axs[1,1].imshow(np.zeros(np.shape(conf_plot)),interpolation='none')
        axs[1,1].title.set_text('Filler')
        axs[1,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,1].set_xlabel('Z Position')
        axs[1,1].set_ylabel('X Position')
        # Wrap up
        plt.show()
