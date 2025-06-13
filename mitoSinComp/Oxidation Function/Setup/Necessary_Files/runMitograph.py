from MitoCode_Functions import open_file_dialog, open_folder_dialog
import numpy as np 
import os

def get_directory(wslcheck, prompt):
    if wslcheck:
        directory = input(prompt)
        if os.path.exists(directory): 
            print("This path exists")
        else: 
            print('Path does not exist')
    else: 
        directory = open_folder_dialog(prompt)
    return directory

def update_default_directory(directory):
    ans = input("Would you like to make this default? (y/n)")
    if ans.lower() == 'y': 
        with open('default_mitocode_mitograph.txt', 'w') as f:
            f.write(directory + '\n')

def runMitograph(wslcheck = True, mitographDirectory = None, structDir = None, dxy = 0.104, dz = 0.5):
    if  (os.path.isfile('default_mitocode_mitograph.txt')) or (mitographDirectory != None):
        with open('default_mitocode_mitograph.txt') as f:
            print('Reading from default:')
            mitographDirectory = f.readline().strip()
            print(mitographDirectory)
    else: 
        print("Please select the directory that has the mitograph executable (likely inside build)")
        mitographDirectory = get_directory(wslcheck, "Type in the directory that has mitograph")
        if os.path.isfile(mitographDirectory):
            update_default_directory(mitographDirectory)
        
        if not os.path.isdir(mitographDirectory):
            print("Directory does not exist!")
            mitographDirectory = get_directory(wslcheck, "Type in the directory that has mitograph")
            update_default_directory(mitographDirectory)

    if structDir == None:
        structDir = get_directory(wslcheck, "Enter the directory that has the analysis files")

    prodir = [structDir + "/" + file +"/channel_0/" for file in os.listdir(structDir) if file.startswith("Analysis")]
    os.chdir(mitographDirectory)
    print("Moved to directory")

    for i in prodir: 
        command = "./MitoGraph -xy {} -z {} -path {} -analyze".format(dxy, dz, i)        
        print('Command: ', command)
        os.system(command)

    print("Finished running Mitograph")
    