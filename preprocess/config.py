####################################################################
############## parameters
# task
#
####### for label
# label_vFn
# label_vAnnFile
#
###### for crop
# crop_vFn
# crop_vAnnFile
# outDirPos
# outDirNeg
# cropSize
#
###### example
#
# cfg = Configuration()
# if cfg.task == 'label':
#   print(cfg.label_vFn, cfg.label_vAnnFile)
# elif cfg.task == 'crop':
#   ...
#####################################################################


class Configuration:
    def __init__(self):
        self.__filePath = "../configure.ini"
        fp_cfg = open(self.__filePath, 'r')
        self.__strConfig = fp_cfg.readlines()
        self.task = ''
        self.__initialize()

    def __initialize(self):
        for line in self.__strConfig:
            line = line.strip()
            if line.find('gl_task') >= 0:
                self.task = line.split('=')[1].strip()
            if self.task == 'train':
                # crop
                if line.find('crop_rect_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.crop_rect_fn = [name.strip() for name in tmp]
                if line.find('crop_fn_video') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.crop_vFn = [name.strip() for name in tmp]
                if line.find('crop_fn_annotation') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.crop_vAnnFile = [name.strip() for name in tmp]
                if line.find('crop_dir_pos') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.outDirPos = [name.strip() for name in tmp]
                if line.find('crop_dir_neg') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.outDirNeg = [name.strip() for name in tmp]
                if line.find('crop_size') >= 0:
                    size = line.split('=')[1].strip().split(',')
                    self.cropSize = [int(i) for i in size]
                if line.find('origin_x_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.origin_x_fn = [name.strip() for name in tmp]
                if line.find('origin_y_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.origin_y_fn = [name.strip() for name in tmp]
                if line.find('train_x_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.train_x_fn = [name.strip() for name in tmp]
                if line.find('train_y_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.train_y_fn = [name.strip() for name in tmp]
                if line.find('valid_x_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.valid_x_fn = [name.strip() for name in tmp]
                if line.find('valid_y_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.valid_y_fn = [name.strip() for name in tmp]
                if line.find('test_x_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.test_x_fn = [name.strip() for name in tmp]
                if line.find('test_y_fn') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.test_y_fn = [name.strip() for name in tmp]
                # hog parameters
                if line.find('hog_win_size') >= 0:
                    tmp = line.split('=')[1].split(',')
                    self.hogWinSize = (int(tmp[0]), int(tmp[1]))
                if line.find('hog_block_size') >= 0:
                    tmp = line.split('=')[1].split(',')
                    self.hogBlockSize = (int(tmp[0]), int(tmp[1]))
                if line.find('hog_block_stride') >= 0:
                    tmp = line.split('=')[1].split(',')
                    self.hogBlockStride = (int(tmp[0]), int(tmp[1]))
                if line.find('hog_cell_size') >= 0:
                    tmp = line.split('=')[1].split(',')
                    self.hogCellSize = (int(tmp[0]), int(tmp[1]))
                if line.find('hog_nbins') >= 0:
                    self.hogNbins = int(line.split('=')[1])
                # MLP
                if line.find('mlp_n_in') >= 0:
                    self.mlp_n_in = eval(line.split('=')[1].strip())
                if line.find('mlp_n_out') >= 0:
                    self.mlp_n_out = int(line.split('=')[1].strip())
                if line.find('mlp_learning_rate') >= 0:
                    self.mlp_learning_rate = float(line.split('=')[1].strip())
                if line.find('mlp_l1_reg') >= 0:
                    self.mlp_l1_reg = float(line.split('=')[1].strip())
                if line.find('mlp_l2_reg') >= 0:
                    self.mlp_l2_reg = float(line.split('=')[1].strip())
                if line.find('mlp_n_epochs') >= 0:
                    self.mlp_n_epochs = int(line.split('=')[1].strip())
                if line.find('mlp_batch_size') >= 0:
                    self.mlp_batch_size = int(line.split('=')[1].strip())
                if line.find('mlp_my_patience') >= 0:
                    self.mlp_patience = int(line.split('=')[1].strip())
                if line.find('mlp_patience_increase') >= 0:
                    self.mlp_patience_increase = int(line.split('=')[1].strip())
                if line.find('mlp_improvement_threshold') >= 0:
                    self.mlp_improvement_threshold = float(line.split('=')[1].strip())
                if line.find('mlp_n_list_hidden_nodes') >= 0:
                    self.mlp_n_list_hidden_nodes = eval(line.split('=')[1].strip())
                # ROC
                if line.find('ROC_title') >= 0:
                    self.roc_title = line.split('=')[1].strip()
                if line.find('ROC_legend') >= 0:
                    self.roc_legend = [item.strip() for item in line.split('=')[1].split(',')]
                if line.find('ROC_save_path') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.rocSavePath = [path.strip() for path in tmp]
                if line.find('ROC_prob') >= 0:
                    tmp = line.split('=')[1].strip().split(',')
                    self.roc_prob = [name.strip() for name in tmp]