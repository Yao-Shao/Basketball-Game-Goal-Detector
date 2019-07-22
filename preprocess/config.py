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
        self.__filePath = "../configure.txt"
        fp_cfg = open(self.__filePath, 'r')
        self.__strConfig = fp_cfg.readlines()
        self.task = ''
        self.__initialize()

    def __initialize(self):
        for line in self.__strConfig:
            line = line.strip()
            if line.find('gl_task') >= 0:
                self.task = line.split('=')[1].strip()
            if self.task == 'label':
                if line.find('label_fn_video') >= 0:
                    self.label_vFn = line.split('=')[1].strip()
                if line.find('label_fn_annotation') >= 0:
                    self.label_vAnnFile = line.split('=')[1].strip()
            if self.task == 'crop':
                if line.find('crop_fn_video') >= 0:
                    self.crop_vFn = line.split('=')[1].strip()
                if line.find('crop_fn_annotation') >= 0:
                    self.crop_vAnnFile = line.split('=')[1].strip()
                if line.find('crop_dir_pos') >= 0:
                    self.outDirPos = line.split('=')[1].strip()
                if line.find('crop_dir_neg') >= 0:
                    self.outDirNeg = line.split('=')[1].strip()
                if line.find('crop_size') >= 0:
                    size = line.split('=')[1].strip()
                    self.cropSize = [int(size.split(',')[0]), int(size.split(',')[1])]
            if self.task == 'train' >= 0:
                pass
            if self.task == 'testROC' >= 0:
                pass


