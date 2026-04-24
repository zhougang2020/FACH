import sys
sys.path.insert(0, 'D:/Mywork/SMC/codes/DeepHash/Cross_Modal_Retrieval/FACH')
from datasets import load_data
imgs, tags, labels = load_data('mirflickr25k', use_vgg_feat=True)
print('images:', imgs.shape, 'tags:', tags.shape, 'labels:', labels.shape)
