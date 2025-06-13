from MitoCode_Functions import *
from Mitocode_Intensity import *
import tifffile
import os
from aicsimageio import AICSImage

class image_splitter():
    def __init__(self, dir):
        self.dir = dir
        self.a = AICSImage(dir)
        self.len_of_channels = self.a.dims['C'][0]
        self.channel_data_list = []
        self.name_of_file = str(dir.split('/')[-1].split('.')[0])

    def make_folders(self):
        stor_dir = os.path.join(os.path.dirname(str(self.dir)), 'AnalysisFiles_{}'.format(self.name_of_file))
        # check if the folder already exists
        try: 
            os.makedirs(stor_dir)
        except: 
            FileExistsError
        for j in range(self.len_of_channels): 
            curr_save_dir = os.path.join(stor_dir, 'channel_{}'.format(j))
            try: 
                os.makedirs(curr_save_dir)
            except: 
                FileExistsError
            tifffile.imwrite(os.path.join(curr_save_dir,  'channel_{}.tif'.format(j)), self.a.get_image_data("ZYX", C=j, S=0, T=0), bigtiff= True)

def splitter_main(directory = ''):
    if directory != '': 
        print(directory)
    else: 
        directory = open_folder_dialog()
    file_list = os.listdir(directory)
    czi_dir = [directory + "/" + file for file in file_list if file.endswith(".czi")]
    tif_dir = [directory + "/" + file for file in file_list if file.endswith(".tif")]
    

    
    print("{} czi files detected".format(len(czi_dir)))
    print("{} tif files detected".format(len(tif_dir)))
    
    dir = czi_dir + tif_dir
    
    for file in dir:
        print(file)
        _splitter = image_splitter(str(file))
        _splitter.make_folders()
    
