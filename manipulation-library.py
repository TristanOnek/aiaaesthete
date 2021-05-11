'''
MANIPULATION LIBRARY
Purpose: Extra image functions that allow for AIA and associated programs to add additional image effects
Projects and Initiatives: AIA/SA/AIB
Status: Experimental, Fragment, Dependent
Creator: T.O.
'''

import numpy as np
from PIL import Image as im
from PIL import ImageFilter as ifl
import random
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

#IMGAUG TEST SUBSECTION:
ia.seed(4)


# This is a testing version of manipulation-library, so we're spinning up random values to generate output
randval1 = random.randint(3, 7)
randval2 = random.randint(3, 7)
randval3 = random.randint(5, 100)
randval4 = random.randint(5, 100)
randval5 = random.randint(200, 250)
randval6 = random.uniform(0, 1)
randval7 = random.uniform(40, 50)

#Working manipulations with input AIA images:
rotate = iaa.Affine(rotate=(-25, 25))

seq = iaa.Sequential([
    iaa.Affine(translate_px={"x":-40}),
    iaa.AdditiveGaussianNoise(scale=0.1*255)
])

emb = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))

warp = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))

jigsaw = iaa.Jigsaw(nb_rows=randval1, nb_cols=randval2)

elastic = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)

uv = iaa.UniformVoronoi(randval5, p_replace=randval6, max_size=None)

rgv = iaa.RegularGridVoronoi(randval5, randval1)

rrgv = iaa.RelativeRegularGridVoronoi(
    (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=512)

pt = iaa.PerspectiveTransform(scale=(0.01, 0.20))


#End manipulations

'''
#Working image_aug executions:
image_aug = seq(image=numpyim)
image_aug = rotate(image=numpyim)
image_aug = emb(image=numpyim)   #doesn't seem noticeable with super distorted input
image_aug = warp(image=numpyim) #i like this one
image_aug = jigsaw(image=numpyim) #wacky jigsaw effects, i like this one too
image_aug = elastic(image=numpyim) #again, not super noticeable
image_aug = uv(image=numpyim) #may be excessive but looks neat
image_aug = rgv(image=numpyim) #this is our minimalist art answer
image_aug = rrgv(image=numpyim) #also good for minimalism
image_aug = pt(image=numpyim) #good for making things wavy
What always works: rgv, jigsaw, rrgv
'''

image = im.open('output_art/finalproduct.png')   #open the image
numpyim = np.array(image)                   #convert to np image type
image_aug = jigsaw(image=numpyim)
#image_aug2 = jigsaw(image=image_aug)
#image_aug = rgv(image=numpyim)
im = im.fromarray(image_aug)
#im = im.filter(ifl.GaussianBlur(radius=randval7))
im.save('finalproduct.png')

print(randval1)
print(randval2)
print(randval3)
print(randval4)
print(randval5)
print(randval6)

print("Augmented:")
ia.imshow(image_aug)
# END TEST SECTION