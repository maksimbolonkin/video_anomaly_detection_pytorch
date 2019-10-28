from PIL import Image
import torch.utils.data as data
import os
import torchvision.transforms as transforms
import torch

class UCSDAnomalyDataset(data.Dataset):
    '''
    Dataset class to load  UCSD Anomaly Detection dataset
    Input: 
    - root_dir -- directory (Train/Test) structured exactly as out-of-the-box folder downloaded from the site
    http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
    - time_stride (default 1) -- max possible time stride used for data augmentation
    - seq_len -- length of the frame sequence
    Output:
    - tensor of 10 normlized grayscale frames stiched together
    
    Note:
    [mean, std] for grayscale pixels is [0.3750352255196134, 0.20129592430286292]
    '''
    def __init__(self, root_dir, seq_len = 10, time_stride=1, transform=None):
        super(UCSDAnomalyDataset, self).__init__()
        self.root_dir = root_dir
        vids = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.samples = []
        for d in vids:
            for t in range(1, time_stride+1):
                for i in range(1, 200):
                    if i+(seq_len-1)*t > 200:
                        break
                    self.samples.append((os.path.join(self.root_dir, d), range(i, i+(seq_len-1)*t+1, t)))
        self.pil_transform = transforms.Compose([
                    transforms.Resize((227, 227)),
                    transforms.Grayscale(),
                    transforms.ToTensor()])
        self.tensor_transform = transforms.Compose([
                    transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))])
        
    def __getitem__(self, index):
        sample = []
        pref = self.samples[index][0]
        for fr in self.samples[index][1]:
            with open(os.path.join(pref, '{0:03d}.tif'.format(fr)), 'rb') as fin:
                frame = Image.open(fin).convert('RGB')
                frame = self.pil_transform(frame) / 255.0
                frame = self.tensor_transform(frame)
                sample.append(frame)
        sample = torch.stack(sample, axis=0)
        return sample

    def __len__(self):
        return len(self.samples)