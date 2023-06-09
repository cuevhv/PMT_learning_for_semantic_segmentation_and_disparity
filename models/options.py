    x = Concatenate()([a_b4, b_b4])
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=64, kernel_size=1)(x)
    x = Conv2DownUp(x, 32, 3, padding)
    x1 = UpSampling2D(2)(x)
    seg_branch = Conv2DownUp(x1, 32, 3, padding, lastLayer=False)
    seg_branch = Conv2DTranspose(filters=labels, kernel_size=3, padding=padding)(seg_branch)
    seg_branch = UpSampling2D(8)(seg_branch)
    seg_branch = Activation('softmax', name='seg_out')(seg_branch)
    
    window_size = int(a_pyramidB_2.shape[2]/1.92)/2
    y = Getcorr1d(a_pyramidB_2, b_pyramidB_2, displacement=window_size, padding=-1)
    y = Conv2D(filters=128, kernel_size=1)(y)
    y1 = Conv2DownUp(x1, 128, 3, padding)
    y = Concatenate()([y1, y])
    y = Conv2DownUp(y, 64, 3, padding)
    disp_out = UpSampling2D(2)(y)
    disp_out = Conv2D(filters=64, kernel_size=1)(disp_out)
    disp_out = Conv2DownUp(disp_out, 64, 5, padding = padding, lastLayer=False)
    disp_out = Conv2DTranspose(filters=1, kernel_size=5, padding=padding)(disp_out)
    disp_out = UpSampling2D(4)(disp_out)

    
    

    x = Concatenate()([a_b4, b_b4])
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=64, kernel_size=1)(x)
    x = Conv2DownUp(x, 32, 3, padding)
    x1 = UpSampling2D(2)(x)
    x = UpSampling2D(16)(x)
    seg_branch = Concatenate()([x, xleft1])
    seg_branch = Conv2D(filters=64, kernel_size=1)(seg_branch)
    seg_branch = Conv2DownUp(seg_branch, 32, 3, padding, lastLayer=False)
    seg_branch = Conv2DTranspose(filters=labels, kernel_size=3, padding=padding)(seg_branch)
    seg_branch = Activation('softmax', name='seg_out')(seg_branch)

    window_size = int(a_pyramidB_2.shape[2]/1.92)/2
    y = Getcorr1d(a_pyramidB_2, b_pyramidB_2, displacement=window_size, padding=-1)
    y = Conv2D(filters=128, kernel_size=1)(y)
    y1 = Conv2DownUp(x1, 128, 3, padding)
    y = Concatenate()([y1, y])
    y = Conv2DownUp(y, 64, 3, padding)
    y2 = UpSampling2D(8)(y)
    disp_out = Concatenate()([y2, xleft2])
    disp_out = Conv2D(filters=64, kernel_size=1)(disp_out)
    disp_out = Conv2DownUp(disp_out, 64, 5, padding = padding, lastLayer=False)
    disp_out = Conv2DTranspose(filters=1, kernel_size=5, padding=padding)(disp_out)
