from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import time
import os
import scipy.io
import math
from tqdm import tqdm
from model import ft_net

def main():
    which_epoch = 'last'
    test_dir = './Market/pytorch'
    name = 'ft_ResNet50'
    batchsize = 256
    linear_num = 512
    ms = '1'
    stride = 2

    nclasses = 751 



    print('We use the scale: %s'%ms)
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # setup CUDA
    torch.cuda.set_device(0)
    cudnn.benchmark = True

    h, w = 256, 128

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_dir = test_dir

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                shuffle=False, num_workers=8) for x in ['gallery','query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = 'model.pth'
        network.load_state_dict(torch.load(save_path, weights_only=True))
        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature(model,dataloaders):
        pbar = tqdm()

        for iter, data in enumerate(dataloaders):
            img, label = data
            n, c, h, w = img.size()
            pbar.update(n)
            ff = torch.FloatTensor(n,linear_num).zero_().cuda()

            for i in range(2): # process each image twice - original and flipped
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    # if scale != 1: # process the image with different scales
                    #     # bicubic is only  available in pytorch>= 1.1
                    #     input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) #normalize the features 
            ff = ff.div(fnorm.expand_as(ff))

            
            if iter == 0: # store the first batch from the starting
                features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            #features = torch.cat((features,ff.data.cpu()), 0)
            start = iter*batchsize # calculate the starting index of the second and so on batches
            end = min( (iter+1)*batchsize, len(dataloaders.dataset))
            features[ start:end, :] = ff
        pbar.close()
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)

    model_structure = ft_net(nclasses, stride = stride, linear_num=linear_num)


    model = load_network(model_structure)

    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()



    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders['gallery'])
        query_feature = extract_feature(model,dataloaders['query'])
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('pytorch_result.mat',result)

    print(name)
    result = 'result.txt'
    os.system('python evaluate.py | tee -a %s'%result)


if __name__ == '__main__':
    main()