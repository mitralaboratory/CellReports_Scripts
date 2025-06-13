import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from MitoCode_Functions import open_file_dialog
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import sys

def plot_3D(df, df_dir): 
   for name_folder in df['folder_name_x'].unique():
      name_folder = str(name_folder)
      sidedf = df[df['folder_name_x'] == name_folder]

      element_df = sidedf[['folder_name_x', 'cc_x', 'line_id','element_length_(um)', 'element_average_width','element_pixel_intensity_488', 'element_pixel_intensity_405',
         'element_pixel_intensity_ratio', 'folder_name_x','element_Volume_Voxel', 'x_x', 'y_x', 'z_x']]   

      cc_df = sidedf[['folder_name_x','cc_x', 'x_x', 'y_x', 'z_x', 'line_id', 'cc_length_(um)', 'cc_vol_from_img_(um3)', 'branches', 'nodes', 'edges', 'cc_pixel_intensity_ratio','cc_pixel_intensity_405', 'cc_pixel_intensity_488', 'cc_average_degree_excludeFreeEnds', 'diameter']]
      node_df = sidedf[['folder_name_x', 'cc_x', 'line_id', 'node', 'degree', 'x_y', 'y_y', 'z_y']].dropna()

      merged = cc_df

      cclist = []

      xlist = []
      ylist = []
      zlist = []
      tlist = []

      nxlist = []
      nylist = []
      nzlist = []
      nclist = []

      num_of_cc = len(merged['cc_x'].unique())
      # remove nan values from getting the max

      # first get the values we are to plot 
      maxt = max(merged['cc_pixel_intensity_ratio'].dropna())
      for i in range(0, num_of_cc):
         merged_cc= merged[merged['cc_x'] == i]
         # repeat this for all the other CCs, and then plot them all on the same graph
         line_ids_in_cc = merged_cc['line_id'].unique()
         line_list = []
         for j in range(len(line_ids_in_cc)):
            merged_line = merged_cc[merged_cc['line_id'] == line_ids_in_cc[j]]
            # nodes_temp = merged_line[merged_line['Node'] > 0]

            x = merged_line['x_x'].to_numpy()
            y = merged_line['y_x'].to_numpy()
            z = merged_line['z_x'].to_numpy()
            t = merged_line['cc_pixel_intensity_ratio'].to_numpy()

            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
            tlist.append(t)



            line_list.append([x, y, z, t])

         cclist.append([line_list, i])


      fig = plt.figure(figsize=(15, 15))
      ax = fig.add_subplot(111, projection='3d')
      # change colormap of ax

      # plot the values 

      for i in range(len(cclist)):
         line_list = cclist[i][0]
         cc_number = cclist[i][1]
         # generate a random color 
         color = np.random.rand(3,)
         addedCC = False
         for j in range(len(line_list)):
            x = line_list[j][0]
            y = line_list[j][1]
            z = line_list[j][2]
            t = line_list[j][3]

            N_points = len(x)

            t /= maxt
            for k in range(1, N_points):
                  ax.plot(x[k-1:k+1], y[k-1:k+1], z[k-1:k+1], c=color, linewidth=3, alpha=0.5)

                  # Add CC number as text next to the plotted point
                  if addedCC == False: 
                     ax.text(x[k], y[k], z[k], str(cc_number), color=color, fontsize=8)
                     addedCC = True

      # plot the nodes, where the color of the node is determined by the degree of the node
      # first get the max degree
      # create a list of 10 colors 
      # suggest another color

      colors = ['white', 'grey', 'black', 'yellow', 'cyan', 'orange', 'pink', 'magenta', 'black', 'purple']
      # now based on the degree, assign a color to the node, based on the color list
      # drop duplicate node_df values
      node_df = node_df.drop_duplicates(subset=['node'])
      nx = node_df['x_y'].to_numpy()
      ny = node_df['y_y'].to_numpy()
      nz = node_df['z_y'].to_numpy()
      deg = node_df['degree'].to_numpy()
      node_name = node_df['node'].to_numpy()
      

      for i in range(len(nx)):
         color_index = int(deg[i]) % 10
         node_number = int(node_name[i])
         ax.scatter(nx[i], ny[i], nz[i], color=colors[color_index], s=6)
         # ax.text(nx[i], ny[i], nz[i], str(node_number), color='black', fontsize=10)

      # Create a legend
      for i, color in enumerate(colors):
         ax.scatter([], [], color=color, s=30, label=f'PK {i}')
      
      ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Degree', loc='upper right')

      ax.view_init(90, 270)
         # make the plot a bit longer
         # add a color bar
      ax.set_aspect('equal')
      # include a legend for the nodes, where each color is the degree of the node
      # do it so legend isl ike this red: 1, blue :2 etc
            
      # plt.show()
      plt.title(merged['folder_name_x'].values[0])
      plt.savefig(os.path.join(os.path.dirname(df_dir),  str(name_folder)) + '_node_3D.png', dpi=300)

   return xlist, ylist, zlist, tlist


def create_brightest_cmap(data):
    # Normalize data to range [0, 1]
    norm = plt.Normalize(min(data), max(data))
    
    # Define colors - brightest is white, darkest is black
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0.5, 0, 0)]
    positions = [0, 0.25, 0.5, 0.75, 1]
    # Create a colormap
    cmap = LinearSegmentedColormap.from_list('brightest', list(zip(positions, colors)))
    
    return cmap
from matplotlib.colors import Normalize

def plot_3D_element_LUT(df, df_dir, version=0): 
   for name_folder in df['folder_name_x'].unique():
      plt.clf()
      sidedf = df[df['folder_name_x'] == name_folder]
      element_df = sidedf[['folder_name_x', 'line_id','element_length_(um)', 'element_average_width','element_pixel_intensity_488', 'element_pixel_intensity_405', 'element_pixel_intensity_ratio', 'folder_name_x','element_Volume_Voxel', 'x_x', 'y_x', 'z_x', 'cc_pixel_intensity_ratio']]   
      node_df = sidedf[['folder_name_x', 'cc_x', 'line_id', 'node', 'degree', 'x_y', 'y_y', 'z_y']].dropna()
      num_of_line_id = len(element_df['line_id'].unique())

      # set color bar as the max and min of the element_pixel_intensity_ratio

      #maxt = max(element_df['element_pixel_intensity_ratio'].dropna())
      #mint = min(element_df['element_pixel_intensity_ratio'].dropna())

      # these are tested values for the population. maxt is the maximum network pixel intensity ratio. 
      maxt = 0.3064  # THIS CAN BE CHANGED ACCORDINGLY
      mint = 0.0043

      line_list = []
      fig = plt.figure(figsize=(15, 15))
      ax = fig.add_subplot(111, projection='3d')
      cmap = plt.get_cmap('jet')

      for i in range(0, num_of_line_id):
         curr_line= element_df[element_df['line_id'] == i]
            # repeat this for all the other CCs, and then plot them all on the same graph

         x = curr_line['x_x'].to_numpy()
         y = curr_line['y_x'].to_numpy()
         z = curr_line['z_x'].to_numpy()
         if version == 0:
            t = curr_line['element_pixel_intensity_ratio'].to_numpy()
         line_list.append([x, y, z, t])
         
      q = len(line_list)

      # norm = Normalize(vmin=0, vmax=0.57)  # Scale the colormap to your data range
      norm = Normalize(vmin=mint, vmax=maxt)  # Scale the colormap to your data range

      for j in range(q):
            x = line_list[j][0]
            y = line_list[j][1]
            z = line_list[j][2]
            t = line_list[j][3]

            # for k in range(1, len(x)):
            #    if t[0] > 0 and t[0] < 0.06:
            #          color = 'blue'
            #    elif t[0] >= 0.06 and t[0] < 0.08:
            #          color = 'blue'
            #    elif t[0] >= 0.08 and t[0] < 0.1:
            #          color = 'blue'
            #    elif t[0] >= 0.1 and t[0] < 0.12:
            #          color = 'teal'
            #    elif t[0] >= 0.12 and t[0] < 0.14:
            #          color = 'lightgreen'
            #    elif t[0] >= 0.14 and t[0] < 0.16:
            #          color = 'yellow'
            #    elif t[0] >= 0.16 and t[0] < 0.18:
            #          color = 'red' 
            #    elif t[0] >= 0.18 :
            #          color = 'red'  
               
            for k in range(1, len(x)):
               ax.plot(x[k-1:k+1], y[k-1:k+1], z[k-1:k+1], c=cmap(norm(t[0])), linewidth=3.5, alpha=0.5)
      colors = ['white', 'grey', 'black', 'yellow', 'cyan', 'orange', 'pink', 'magenta', 'black', 'purple']
      # now based on the degree, assign a color to the node, based on the color list
      # drop duplicate node_df values
      node_df = node_df.drop_duplicates(subset=['node'])
      nx = node_df['x_y'].to_numpy()
      ny = node_df['y_y'].to_numpy()
      nz = node_df['z_y'].to_numpy()
      node_number = node_df['node'].to_numpy()
      deg = node_df['degree'].to_numpy()
      

      for i in range(len(nx)):
         color_index = int(deg[i]) % 10
         ax.scatter(nx[i], ny[i], nz[i], color='gray', alpha = 0.4, s=1)
         # ax.text(nx[i], ny[i], nz[i], str(node_number), color='black', fontsize=1)
      
      # # Create a legend
      # for i, color in enumerate(colors):
      #    ax.scatter([], [], color=color, s=30, label=f'PK {i}')
      ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Degree', loc='upper right')

      #    ax.scatter(x[i], y[i], z[i], c=c_value[i], s=10, alpha=0.5)
      ax.view_init(90, 270)
      # make the plot a bit longer
      # add a color bar
      # cmap = plt.cm.jet
      sm = plt.cm.ScalarMappable(cmap=cmap)
      sm.set_array([])
      cbar = fig.colorbar(sm, ax=ax, label='Color Bar', shrink=0.4)
      sm.set_clim(mint, maxt)  # Set vmin and vmax
   
      ax.set_aspect('equal')
         # plt.title(merged['folder_name'][0])
                  # plt.title(merged['folder_name'][0])
      plt.savefig(os.path.join(os.path.dirname(df_dir),  str(name_folder)) + '_LUT.png', dpi=300)
      # plt.savefig(os.path.join(os.path.dirname(df_dir),  df['folder_name_x'].unique()[0] + '_LUT.png'), dpi=300)


# /home.
print("This script will generate 3D graphs for the data in the excel file. Select the excel file.")
df_dir = open_file_dialog('Select Full Table') 

# remove the last part of the path
# read the csv file 
df = pd.read_csv(df_dir[0], index_col=0)

plot_3D(df, df_dir[0])
# plot_LUT(xlist, ylist, zlist, tlist, df_dir[0])
plot_3D_element_LUT(df, df_dir[0])  
