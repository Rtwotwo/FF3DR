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
OPENMVS_BIN = os.path.join(current_path, 'RelWithDebInfo')


class DSM_from_Mesh:
    def __init__(self, input_path, output_path, dsm_uint, dsm_size):
        self.OPENMVS_BIN = OPENMVS_BIN
        self.input_path = input_path
        self.output_path = output_path
        self.dsm_uint = dsm_uint
        self.dsm_size = dsm_size

    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)


    def create_dsm(self, workspace, strInputFileName, strOutputFileName, bbx_border, dsm_name):
        cmd = [os.path.join(self.OPENMVS_BIN, "DensifyPointCloud"),
               "-i", strInputFileName,
               "-o", strOutputFileName,
               "-w", workspace,
               "--dsm_height", str(self.dsm_size[1]), "--dsm_width", str(self.dsm_size[0]),
               "--dsm_unitx", str(self.dsm_uint[0]), "--dsm_unity", str(self.dsm_uint[1]),
               "--dsm_startx", str(bbx_border[0]), "--dsm_endx", str(bbx_border[1]),
               "--dsm_starty", str(bbx_border[2]), "--dsm_endy", str(bbx_border[3]),
               "--dsm_name", dsm_name]

        print(cmd)

        pIntrisics = subprocess.Popen(cmd)
        pIntrisics.wait()


    def create(self, bbx_border, dsm_name='0_0'):

        self.create_folder(self.output_path)
        t1 = time.time()

        print("\n-----------> begin to generate dsm from mesh")
        self.create_dsm(self.output_path, self.input_path, self.output_path, bbx_border, dsm_name)

        t5 = time.time()
        print("---------------Cost {:.4f} min-------------".format((t5 - t1) / 60.0))


parser = argparse.ArgumentParser(description='generate dsm and dom from mesh (.ply)')
parser.add_argument('--mesh_path', type=str, default=r'G:\PCGSPRO_1715406050\whu_omvs_test\virtual\models\pc\0\transfer_mesh')
parser.add_argument('--mesh_format', type=str, default="ply", help="ply/bin/las")
parser.add_argument('--out_dsm_path', type=str, default=r'G:\PCGSPRO_1715406050\whu_omvs_test\virtual\models\pc\0\DSM_from_Mesh')

parser.add_argument('--dsm_uint', default=[0.2, 0.2], help='set dsm resolution')
parser.add_argument('--dsm_size', default=[2900, 2900], help='set dsm size')
parser.add_argument('--bbx_border', default=[-430.0, 150.0, -330.0, 250.0, 700.0, 900.0], help='set dsm border[x_min, x_max, y_min, y_max, z_min, z_max]')

args = parser.parse_args()


if __name__ == '__main__':

    start = time.time()
    print("============== Create DSM Start! =============")
    input_path = args.mesh_path
    input_format = args.mesh_format
    output_path = args.out_dsm_path
    file_name = '0_0'

    dsm_uint = args.dsm_uint
    dsm_size = args.dsm_size
    bbx_border = args.bbx_border

    dm = DSM_from_Mesh(input_path, output_path, dsm_uint, dsm_size)
    dm.create(bbx_border, dsm_name=file_name)

    end = time.time()
    print("---------------Cost {:.4f} min-------------".format((end - start) / 60.0))
    print("============ Create DSM Finished! ===========")



