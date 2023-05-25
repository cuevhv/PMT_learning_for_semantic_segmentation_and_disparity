def GetDirFromText(file_name):
    if "kfold" in file_name:
        main_path = file_name.rsplit('/kfold', 1)[0] + '/'
    else:
        main_path = file_name.rsplit('/', 1)[0] + '/'
    return [main_path + line.rstrip('\n') for line in open(file_name)]


def getTextDataset(CFG):
    # Values only compulsory for training, not eval
    colorL, colorR, disp, seg, inst = [], [], [], [], []
    if CFG.train:
        colorL = GetDirFromText(CFG.colorL)
        colorR = GetDirFromText(CFG.colorR)
        disp = GetDirFromText(CFG.disp)
        seg = GetDirFromText(CFG.seg)
        inst = GetDirFromText(CFG.inst) if (CFG.datasetName != 'garden' and CFG.datasetName != 'roses') else seg
    colorL_test = GetDirFromText(CFG.colorL_test)
    colorR_test = GetDirFromText(CFG.colorR_test)
    disp_test = GetDirFromText(CFG.disp_test)
    seg_test = GetDirFromText(CFG.seg_test)
    inst_test = GetDirFromText(CFG.inst_test) if (CFG.datasetName != 'garden' and CFG.datasetName != 'roses') else seg_test
    return colorL, colorR, disp, seg, inst, colorL_test, colorR_test, disp_test, seg_test, inst_test
