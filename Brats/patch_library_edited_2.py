import numpy as np
import random
import os
from glob import glob
import matplotlib
import skimage
#import matplotlib.pyplot as plt
from skimage import *
from skimage.io import imread,imsave
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.color import rgb2gray
#import sklearn

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)


class PatchLibrary(object):
    def __init__(self, patch_size, train_data, num_samples):
        '''
        class for creating patches and subpatches from training data to use as input for segmentation models.
        INPUT   (1) tuple 'patch_size': size (in voxels) of patches to extract. Use (33,33) for sequential model
                (2) list 'train_data': list of filepaths to all training data saved as pngs. images should have shape (5*240,240)
                (3) int 'num_samples': the number of patches to collect from training data.
        '''

        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        '''
	#Helper function for sampling slices with evenly distributed classes
        #:param class_num: class to sample from choice of {0, 1, 2, 3, 4}.
        #:param num_patches: number of patches to extract
        #:return: num_samples patches from class 'class_num' randomly selected.
        '''
        h, w = self.patch_size[0], self.patch_size[1]
        #patches, labels = [], np.full(num_patches * self.augmentation_multiplier, class_num, 'float')
        patches, labels = [], np.full(int(num_patches) , class_num, 'float')
        print('Finding patches of class {}...'.format(class_num))

        full = False
        start_value_extraction = 0
	print("os path = ",os.path)
        if os.path.isdir('patches/') and os.path.isdir('patches/class_{}/'.format(class_num)):

            # load all patches
            # check if quantity is enough to work
            #path_to_patches = sorted(glob('./patches/class_{}/**.png'.format(class_num)),key=get_right_order)
            path_to_patches = glob('./patches/class_{}/**.png'.format(class_num))
            for path_index in range(len(path_to_patches)):
                if path_index < num_patches:
                    patch_to_add = rgb2gray(imread(path_to_patches[path_index],
                                                   dtype=float)).reshape(4,
                                                                         self.patch_size[0],
                                                                         self.patch_size[1])

                    for el in range(len(patch_to_add)):
                        if np.max(patch_to_add[el]) > 1:
                            patch_to_add[el] = patch_to_add[el] / np.max(patch_to_add[el])

                    patches.append(patch_to_add)
                    print('*---> patch {} loaded and added '.format(path_index))
                else:
                    full = True
                    break

            if len(path_to_patches) < num_patches:
                # change start_value_extraction
                start_value_extraction = len(path_to_patches)
            else:
                full = True
        else:
            os.makedirs('patches/class_{}'.format(class_num))
        if not full:
            ct = start_value_extraction
            while ct < num_patches:
                print('searching for patch {}...'.format(ct))
                im_path = random.choice(self.train_data)
                fn = os.path.basename(im_path)
                label = np.array(imread('Labels/' + fn[:-4] + 'L.png'))
                
                # resample if class_num not in selected slice
                unique, counts = np.unique(label, return_counts=True)
                labels_unique = dict(zip(unique, counts))
                try:
                    if labels_unique[class_num] < 10:
                        continue
                except:
                    print("hi")
                    continue
                # select centerpix (p) and patch (p_ix)
                #img = imread(im_path).reshape(5, 216, 160)[:-1].astype('float')
                img = imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
                p = random.choice(np.argwhere(label == class_num))
                #p_ix = (p[0] - (h / 2), p[0] + ((h + 1) / 2), p[1] - (w / 2), p[1] + ((w + 1) / 2))
                p_ix = (int(p[0]-(h/2)), int(p[0]+((h+1)/2)), int(p[1]-(w/2)), int(p[1]+((w+1)/2)))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                
                # resample if patch is empty or too close to edge
                if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (3 * h * w):
                    if class_num == 0 and patch.shape == (4, h, w):
                        pass
                    else:
                        continue

                for slice_el in range(len(patch)):
                    if np.max(patch[slice_el]) != 0:
                        patch[slice_el] /= np.max(patch[slice_el])
                imsave('./patches/class_{}/{}.png'.format(class_num,
                                                          ct),
                       (np.array(patch.reshape((4 * self.patch_size[0], self.patch_size[1])))))
                patches.append(patch)
                print('*---> patch {} saved and added'.format(ct))
                ct += 1

        print()

        # for i in range(len(patches)):
        #     print('*' * 20)
        #     print('*' * 20)
        #
        #     print(patches[i][0].max())
        #     print(patches[i][0].min())
        #     print('*' * 20)
        #     print('*' * 20)


#        if self.augmentation_angle != 0:
#            print('_*_*_*_ proceed with data augmentation  for class {} _*_*_*_'.format(class_num))
#            print()
#
#            if isdir('.\\patches\\class_{}\\rotations'.format(
#                    class_num)):
#                print("rotations folder present ")
#            else:
#                mkdir_p('.\\patches\\class_{}\\rotations'.format(
#                    class_num))
#                print("rotations folder created")
#            for el_index in range(len(patches)):
#                for j in range(1, self.augmentation_multiplier):
#                    try:
#                        patch_rotated = np.array(rgb2gray(imread('.\\patches\\class_{}\\'
#                                                                 'rotations\\{}_{}.png'.format(class_num,
#                                                                                              el_index,
#                                                                                              self.augmentation_angle * j)),
#                                                          dtype=float)).reshape(4,
#                                                                                self.patch_size[0],
#                                                                                self.patch_size[1])
#
#                        for slice_el in range(len(patch_rotated)):
#                            if np.max(patch_rotated[slice_el]) > 1:
#                                patch_rotated[slice_el] /= np.max(patch_rotated[slice_el])
#
#                        patches.append(patch_rotated)
#                        print('*---> patch {} loaded and added '
#                              'with rotation of {} degrees'.format(el_index,
#                                                                   self.augmentation_angle * j))
#                    except:
#
#                        final_rotated_patch = rotate_patch(np.array(patches[el_index]), self.augmentation_angle * j)
#                        patches.append(final_rotated_patch)
#                        imsave('.\\patches\\class_{}\\'
#                               'rotations\\{}_{}.png'.format(class_num,
#                                                            el_index,
#                                                            self.augmentation_angle * j),
#                               np.array(final_rotated_patch).reshape(4 * self.patch_size[0], self.patch_size[1]))
#                        print(('*---> patch {} saved and added '
#                               'with rotation of {} degrees '.format(el_index,
#                                                                     self.augmentation_angle * j)))
#            print()
#            print('augmentation done \n')

            # for patch in patches:
            #     for i in range(1, self.augmentation_multiplier):
            #         patch_rotate = rotate_patch(patch, self.augmentation_angle * i)
            #         patches.append(patch_rotate)
            # print('data augmentation complete')
            # print()

        return np.array(patches), labels	

    def center_n(self, n, patches):
        '''
        Takes list of patches and returns center nxn for each patch. Use as input for cascaded architectures.
        INPUT   (1) int 'n': size of center patch to take (square)
                (2) list 'patches': list of patches to take subpatch of
        OUTPUT: list of center nxn patches.
        '''
        sub_patches = []
        for mode in patches:
            subs = np.array([patch[(self.h/2) - (n/2):(self.h/2) + ((n+1)/2),(self.w/2) - (n/2):(self.w/2) + ((n+1)/2)] for patch in mode])
            sub_patches.append(subs)
        return np.array(sub_patches)

    def slice_to_patches(self, filename):
        '''
        Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
        INPUT: str 'filename': path to image to be converted to patches
        OUTPUT: list of patched version of imput image.
        '''
        slices = io.imread(filename).astype('float').reshape(5,240,240)[:-1]
        plist=[]
        for slice in slices:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (h,w))
            plist.append(p)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def patches_by_entropy(self, num_patches):
        '''
        Finds high-entropy patches based on label, allows net to learn borders more effectively.
        INPUT: int 'num_patches': defaults to num_samples, enter in quantity it using in conjunction with randomly sampled patches.
        OUTPUT: list of patches (num_patches, 4, h, w) selected by highest entropy
        '''
        patches, labels = [], []
        ct = 0
        while ct < num_patches:
            im_path = random.choice(training_images)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # pick again if slice is only background
            if len(np.unique(label)) == 1:
                continue

            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            l_ent = entropy(label, disk(self.h))
            top_ent = np.percentile(l_ent, 90)

            # restart if 80th entropy percentile = 0
            if top_ent == 0:
                continue

            highest = np.argwhere(l_ent >= top_ent)
            p_s = random.sample(highest, 3)
            for p in p_s:
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                # exclude any patches that are too small
                if np.shape(patch) != (4,65,65):
                    continue
                patches.append(patch)
                labels.append(label[p[0],p[1]])
            ct += 1
            return np.array(patches[:num_samples]), np.array(labels[:num_samples])

    def make_training_patches(self, entropy=False, balanced_classes=True, classes=[0,1,2,3,4]):
        '''
        Creates X and y for training CNN
        INPUT   (1) bool 'entropy': if True, half of the patches are chosen based on highest entropy area. defaults to False.
                (2) bool 'balanced classes': if True, will produce an equal number of each class from the randomly chosen samples
                (3) list 'classes': list of classes to sample from. Only change default oif entropy is False and balanced_classes is True
        OUTPUT  (1) X: patches (num_samples, 4_chan, h, w)
                (2) y: labels (num_samples,)
        '''
        if balanced_classes:
            per_class = self.num_samples / len(classes)
            patches, labels = [], []
            progress.currval = 0
            for i in progress(range(len(classes))):
                p, l = self.find_patches(classes[i], per_class)
                # set 0 <= pix intensity <= 1
                for img_ix in range(len(p)):
                    for slice in range(len(p[img_ix])):
                        if np.max(p[img_ix][slice]) != 0:
                            p[img_ix][slice] /= np.max(p[img_ix][slice])
                patches.append(p)
                labels.append(l)
            return np.array(patches).reshape(self.num_samples, 4, self.h, self.w), np.array(labels).reshape(self.num_samples)

        else:
            print ("Use balanced classes, random won't work.")



if __name__ == '__main__':
    import os
    os.chdir("/home/ec2-user/cardinal/BRATS2015/")
    pass
