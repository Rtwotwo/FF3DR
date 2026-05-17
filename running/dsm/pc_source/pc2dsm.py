# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import os
import subprocess
import yaml
import argparse
import time
import shutil

# Indicate the lib directory
current_path = os.path.dirname(os.path.abspath(__file__))
GRID_BIN = os.path.join(current_path, 'COTIGridInterpolationV2')


class DSM_from_PC:
    def __init__(self, input_path, output_path, dsm_uint, dsm_size, select_method="Robust_Max", interpolation_method=None):
        """
        generate dsm from point cloud
        """
        self.GRID_BIN = GRID_BIN
        self.input_path = input_path
        self.output_path = output_path + '/dsm'
        self.dsm_uint = dsm_uint
        self.dsm_size = dsm_size
        self.select_method = select_method
        self.interpolation_method = interpolation_method

    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)


    def create_dsm(self, workspace, strInputFileName, strOutputFileName, bbx_border):

        cmd = [os.path.join(self.GRID_BIN, "COTIGridInterpolationEXE"),
               "-input_root", strInputFileName, "-out_raster", strOutputFileName,
               "-log_file", workspace,
               "-bin_select_method", str(self.select_method),
               "-interpolation_method", str(self.interpolation_method),
               "--x_start", str(bbx_border[0]), "--y_start",  str(bbx_border[3]),
               "--x_cell", str(self.dsm_uint[0]), "--y_cell", str(self.dsm_uint[1]),
               "--x_size", str(self.dsm_size[0]), "--y_size", str(self.dsm_size[1])]

        pIntrisics = subprocess.Popen(cmd)
        pIntrisics.wait()


    def create(self, bbx_border, dsm_name='0_0'):

        self.create_folder(self.output_path)
        t1 = time.time()

        print("\n-----------> begin to generate dsm from point cloud")
        output_file_name = self.output_path + "/{}.tif".format(dsm_name)
        self.create_dsm(self.output_path, self.input_path, output_file_name, bbx_border)

        t5 = time.time()
        print("---------------Cost {:.4f} min-------------".format((t5 - t1) / 60.0))



parser = argparse.ArgumentParser(description='generate dsm and dom from point cloud (.ply)')
parser.add_argument('--pc_path', type=str, default=r'G:\other_data\pipeline\workspace_tianjin_2_scale4\production_casred_tianjin_0.0397\pc')
parser.add_argument('--pc_format', type=str, default="ply", help="ply/bin/las")
parser.add_argument('--out_dsm_path', type=str, default=r'G:\other_data\pipeline\workspace_tianjin_2_scale4\production_casred_tianjin_0.0397\pc')

parser.add_argument('--dsm_uint', default=[0.2, 0.2], help='set dsm resolution')
parser.add_argument('--dsm_size', default=[2900, 2900], help='set dsm size')
parser.add_argument('--bbx_border', default=[-430.0, 150.0, -330.0, 250.0, 700.0, 900.0], help='set dsm border[x_min, x_max, y_min, y_max, z_min, z_max]')

parser.add_argument('--select_method', default="Robust_Max", help='set dsm generation method')
parser.add_argument('--interpolation_method', default=None, help='set dsm interpolation method')

args = parser.parse_args()


if __name__ == '__main__':

    start = time.time()
    print("============== Create DSM Start! =============")
    input_path = args.pc_path
    input_format = args.pc_format
    output_path = args.out_dsm_path
    file_name = '0_0'

    dsm_uint = args.dsm_uint
    dsm_size = args.dsm_size
    bbx_border = args.bbx_border

    pc_select_method = args.select_method
    pc_interpolation_method = args.interpolation_method

    cdd = DSM_from_PC(input_path, output_path, dsm_uint, dsm_size, pc_select_method, pc_interpolation_method)
    cdd.create(bbx_border, dsm_name=file_name)

    end = time.time()
    print("---------------Cost {:.4f} min-------------".format((end - start) / 60.0))
    print("============ Create DSM Finished! ===========")



