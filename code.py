def load_label_map_cityscapes():

    filler = -1000

    label_colors = torch.FloatTensor([
        [128, 64,128], # road
        [244, 35,232], # sidewalk
        [70, 70, 70 ],  # building
        [102,102,156], # wall
        [190,153,153], # fence
        [153,153,153], # pole
        [250,170, 30], # traffic light
        [220,220,  0], # traffic sign
        [107,142, 35], # vegetation
        [152,251,152], # terrain
        [ 70,130,180], # sky
        [220, 20, 60], # person
        [255,  0,  0], # rider
        [  0,  0,142], # car
        [  0,  0, 70], # truck
        [  0, 60,100], # bus
        [  0, 80,100], # train
        [  0,  0,230], # motorcycle
        [119, 11, 32]  # bicycle	
        ]) 
    return label_colors

def image_to_labels_cityscapes(img):

    img = img.float().mul(255)

    # load the labels
    label_colors = load_label_map_cityscapes()
    Nlabels = label_colors.size()

    label_colors_img = torch.FloatTensor(Nlabels,img.size(0),img.size(1),img.size(2))
    for i in range(0,label_colors_img.size(0)-1) :
            # fill in three color channels
            label_colors_img[i][0].fill(label_colors[i][0])
            label_colors_img[i][1].fill(label_colors[i][1])
            label_colors_img[i][2].fill(label_colors[i][2])

    # assign label that is closest in color for each output pixel
    dists = torch.FloatTensor(label_colors_img.size(0),img.size(1),img.size(2))
    for i in range(0,label_colors_img.size(0)-1) :
            dists[i] = torch.add(img,torch.mul(label_colors_img[i],-1)).pow(2).sum(1).squeeze()

    y, ii = torch.min(dists, 1)
    ii = ii[0]
    ii = ii-1
    ii = ii.float().div(255)

    return ii
