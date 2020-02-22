# Build CNN to classify *Parus domesticus* and *Parus montanus* individuals
> summary


# Passer-classifier

The goal of this project is to create classifier that can separate individuals between two sparrow species: *Passer montanus* and *Passer domesticus*. I intend to use only male individuals here, but the code is extendable for female individuals also, but that might be less performant than the one with only male individuals.

I aim to do this project quick and dirty using fastai's awesome library that runs on top of PyTorch. I love the philosophy behind the fastai library and I intend to use it in the future as well. Let's start by importing the main components from fastai that we need for creating the classifier.

Oh, btw, my approach here is to use CNN based on popular CNN architectures such as ResNet

1. TOC
{:toc}

```python
from fastai.vision import *
```

Before I proceed, I'd like to check what kind of device I'm working with before training the model. As of writing this, I'm using my own PC with RTX 2080 GPU, as we should see here

```python
import torch
torch.cuda.get_device_name(0)
```




    'GeForce RTX 2080'



# Download images for classifier

## Get a list of URLs

To start this process fire up the Google Image search and type in the search words you're after. In my case I typed this into search field:

`"passer montanus" -female`

Most of the search results look pretty ok to me, but don't worry, we'll come back to clean up the dataset a bit later on. As a side note, you can see maximum number of 700 pictures in search, so that is the upper limit for our dataset size for each label. I intend to work with a bit smaller dataset, let's say 400 images for each Google Image search



### Download into file

Now we run some Javascript code in browser which will save the URLs of all the images we want for our dataset. Fire up the developer tool in your browser and paste this code there:

`urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));`

Make sure that you disable your adblocker before running the above code. Running the code downloads the file to your downloads folder

### Create a directory and upload urls file into your machine

```python
folder = 'passer-montanus'
file = 'passer_montanus.csv'
```

```python
folder = 'passer-domesticus'
file = 'passer_domesticus.csv'
```

Then run this cell once per each category

```python
path = Path('data/birds')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```

```python
path.ls()
```




    [PosixPath('data/birds/models'),
     PosixPath('data/birds/passer-domesticus'),
     PosixPath('data/birds/passer_domesticus.csv'),
     PosixPath('data/birds/passer-montanus'),
     PosixPath('data/birds/passer_montanus.csv')]



## Download images

fast.ai has a function that downloads files from respective urls. We just have to specify the urls filename as well as the destination folder and this function will download all images that can be opened. If they have some problem in being opened, they will not be save

Run this line once for every category

```python
classes = ['passer-montanus', 'passer-domesticus']
```

```python
download_images(path/file, dest, max_pics=400)
```

Remove images that can't be opened:

```python
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
```

## View data

```python
np.random.seed(2020)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                 ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
```

Take a look at the classes and pictures

```python
data.classes
```




    ['passer-domesticus', 'passer-montanus']



```python
data.show_batch(rows=3, figsize=(7,8))
```


![png](images/Passer-classifier_files/output_27_0.png)


Here we see below:
- Class labels
- Number of labels
- Training set size
- Validation set size

```python
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```




    (['passer-domesticus', 'passer-montanus'], 2, 512, 128)



# Train model

We use pretrained ResNet model and with fast.ai this is super simple and doesn't take that many lines to train the data. Note here, that we don't need to write CNN architecture because we are using pretrained ResNet model with architecture defined by that model. We can use different models than ResNet, but ResNet34 feels like a good starting point here. If our model over- or underfits significantly we can then think about different models

```python
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
```

```python
learn.fit_one_cycle(5)
```


Total time: 00:10 <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.870170</td>
      <td>0.573786</td>
      <td>0.281250</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.736060</td>
      <td>0.489103</td>
      <td>0.218750</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.636997</td>
      <td>0.497782</td>
      <td>0.257812</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.569070</td>
      <td>0.493976</td>
      <td>0.242188</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.529666</td>
      <td>0.492206</td>
      <td>0.234375</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


Pretty promising results so far with couple of lines with code. At this moment based on those metrics we don't significantly over- or underfit and our model is doing ok (77%). Let's save our model with simple command:

```python
learn.save('stage-1')
```

Now we make all layers available for training by unfreezing the layers making it possible to update weights to all layers

```python
learn.unfreeze()
```

Find the best learning rate for underlying data:

```python
learn.lr_find()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.


```python
learn.recorder.plot()
```


![png](Passer-classifier_files/output_40_0.png)


On the learning rate finder, we are looking for the strongest downward slope that's kind of sticking around for quite a while. This is something you just need a practice on - practice makes perfect!

```python
learn.fit_one_cycle(6, max_lr=slice(3e-4, 3e-3))
```


Total time: 00:13 <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.429921</td>
      <td>0.576350</td>
      <td>0.226562</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.418544</td>
      <td>0.826569</td>
      <td>0.250000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.418443</td>
      <td>1.693962</td>
      <td>0.367188</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.361307</td>
      <td>0.747831</td>
      <td>0.164062</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.312527</td>
      <td>0.577680</td>
      <td>0.156250</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.268261</td>
      <td>0.466389</td>
      <td>0.117188</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


We gained a bit better classifier here as our error_rate went down from 23% to around 11%. Slight improvement, but I'll take it anyway! Let's save this classifier

```python
learn.save('stage-2')
```

## Interpretation

Let's plot simple confusion matrix to see how our model performed

```python
learn.load('stage-2');
```

```python
interp = ClassificationInterpretation.from_learner(learn)
```

```python
interp.plot_confusion_matrix()
```


![png](Passer-classifier_files/output_48_0.png)


Let's see some examples what labels our classifier puts on iamges

```python
learn.show_results()
```


![png](Passer-classifier_files/output_50_0.png)


Based on these results seems like our Passer classifier is doing pretty well. One thing to note here that it seems to be able to classify female individuals of both species in this small sample I showed you above. But you can easily see that our dataset is still pretty corrupted as it contains lots of bad pictures of our Passer individuals. We aim to work with that in the next section

# Cleaning up

Some of our top losses might not be because of our classifier. Most of the problems with top losses come from poor pictures. We aim to remedy this here by pruning our dataset to be a bit better. One way to do that is to use `ImageCleaner` widget from `fastai.widgets` 

```python
from fastai.widgets import *
```

One neat feature of `ImageClearner` widget is that it actually doesn't remove any pictures from folders, it just creates a new csv called `cleaned.csv` from where you can create new `ImageDataBunch` with the corrected labels

We want to perform the cleaning to the whole dataset, so we need to create new dataset without the split:

```python
db = (ImageList.from_folder(path)
     .split_none()
     .label_from_folder()
     .transform(get_transforms(), size=224)
     .databunch())
```

Now we create new `cnn_learner` and load the model we saved after finding the optimal learning rate earlier. We can see the parameters of the model and examine them. We use this new classifier with all the images

```python
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2')
```




    Learner(data=ImageDataBunch;
    
    Train: LabelList (640 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    passer-domesticus,passer-domesticus,passer-domesticus,passer-domesticus,passer-domesticus
    Path: data/birds;
    
    Valid: LabelList (0 items)
    x: ImageList
    
    y: CategoryList
    
    Path: data/birds;
    
    Test: None, model=Sequential(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (4): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (5): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): Sequential(
        (0): AdaptiveConcatPool2d(
          (ap): AdaptiveAvgPool2d(output_size=1)
          (mp): AdaptiveMaxPool2d(output_size=1)
        )
        (1): Flatten()
        (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout(p=0.25)
        (4): Linear(in_features=1024, out_features=512, bias=True)
        (5): ReLU(inplace)
        (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout(p=0.5)
        (8): Linear(in_features=512, out_features=2, bias=True)
      )
    ), opt_func=functools.partial(<class 'torch.optim.adam.Adam'>, betas=(0.9, 0.99)), loss_func=FlattenedLoss of CrossEntropyLoss(), metrics=[<function error_rate at 0x7f94be277950>], true_wd=True, bn_wd=True, wd=0.01, train_bn=True, path=PosixPath('data/birds'), model_dir='models', callback_fns=[functools.partial(<class 'fastai.basic_train.Recorder'>, add_time=True, silent=False)], callbacks=[], layer_groups=[Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU(inplace)
      (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): ReLU(inplace)
      (12): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (13): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (14): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (15): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace)
      (17): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (18): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (19): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (20): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (21): ReLU(inplace)
      (22): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (23): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (24): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (25): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (27): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (28): ReLU(inplace)
      (29): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (30): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (31): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (32): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (33): ReLU(inplace)
      (34): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (35): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (36): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (37): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (38): ReLU(inplace)
      (39): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (40): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ), Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace)
      (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (11): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (13): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (14): ReLU(inplace)
      (15): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (16): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (19): ReLU(inplace)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (24): ReLU(inplace)
      (25): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (27): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (28): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (29): ReLU(inplace)
      (30): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (31): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (32): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (33): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (34): ReLU(inplace)
      (35): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (36): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (37): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (39): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (40): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (41): ReLU(inplace)
      (42): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (43): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (44): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (45): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (46): ReLU(inplace)
      (47): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (48): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ), Sequential(
      (0): AdaptiveAvgPool2d(output_size=1)
      (1): AdaptiveMaxPool2d(output_size=1)
      (2): Flatten()
      (3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): Dropout(p=0.25)
      (5): Linear(in_features=1024, out_features=512, bias=True)
      (6): ReLU(inplace)
      (7): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): Dropout(p=0.5)
      (9): Linear(in_features=512, out_features=2, bias=True)
    )], add_time=True, silent=False)



```python
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
```

```python
ImageCleaner(ds, idxs, path)
```


    'No images to show :)'


{% include screenshot url="passer-classifier/image-cleaner.png" %}

You can see from above screenshot how the `ImageCleaner` looks like in action. We get a batch of pictures with top losses and we get to choose if we want to remove pictures that our models is going to see. This makes it easy to modify and improve our dataset on the fly. Remember that we don't remove these pictures, we just add modifications to `cleaned.csv` and read only the pictures in it (we don't include images we chose to delete in `clean.csv` file).

Once the `ImageCleaner` is done, we get notification: `'No images to show :)`

We can also remove duplicates from the dataste. In order to do this, we need to get the similar pictures using `.from_similars` and then we can run the `ImageCleaner` with `duplicates=True`. This works same way as before

```python
ds, idxs = DatasetFormatter().from_similars(learn_cln)
```

    Getting activations...




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='10' class='' max='10', style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [10/10 00:01<00:00]
</div>



    Computing similarities...


```python
ImageCleaner(ds, idxs, path, duplicates=True)
```


    'No images to show :). 4 pairs were skipped since at least one of the images was deleted by the user.'


![](../../../../../../images/passer-classifier/image-cleaner.png)

Whew, that was a lot of clicking. But now its done and I have a good feeling that our dataset is much better than before. Now the important thing to remember here, is that we have to recreate our `ImageDataBunch` from `cleaned.csv` so that our modifications don't go to waste. So let's do that here

```python
np.random.seed(2020)
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
                              ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
```

Let's do some exploration on our improved dataset and compare the structure to our previous dataset without modifications:

```python
data.classes
```




    ['passer-domesticus', 'passer-montanus']



```python
data.show_batch(rows=3, figsize=(7,8))
```


![png](Passer-classifier_files/output_71_0.png)


```python
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```




    (['passer-domesticus', 'passer-montanus'], 2, 506, 126)



Comparison to our previous dataset (old, new):
- `data.classes` and `data.c`: exactly the same as they should be
- `len(data.train_ds)`: 512 vs. 506
- `len(data.valid_ds)`: 128 vs. 126

So we removed some or our pictures but not that many considering the total number of pictures. Let's see how this affects our model

```python
learn_cleaned_set = cnn_learner(data, models.resnet34, metrics=error_rate)
learn_cleaned_set.load('stage-2');
```

```python
learn_cleaned_set.fit_one_cycle(6, max_lr=slice(3e-4, 3e-3))
```


Total time: 00:11 <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.163263</td>
      <td>0.240788</td>
      <td>0.047619</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.146407</td>
      <td>0.248486</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.133747</td>
      <td>0.287110</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.118506</td>
      <td>0.308891</td>
      <td>0.071429</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.110285</td>
      <td>0.317763</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.107746</td>
      <td>0.302006</td>
      <td>0.071429</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


```python
learn_cleaned_set.save('stage-3')
```

Looks good! We managed to our error rate down to less than 10% with 6 epochs. We might run a bit more epochs to see if our error_rate would go down even more and if our train_loss and valid_loss should be more similar. Let's try that:

```python
learner = cnn_learner(data, models.resnet34, metrics=error_rate)
learner.load('stage-2');
```

```python
learner.fit_one_cycle(10, max_lr=slice(3e-4, 3e-3))
```


Total time: 00:17 <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157103</td>
      <td>0.237543</td>
      <td>0.047619</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.141042</td>
      <td>0.256203</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.164854</td>
      <td>0.250816</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151628</td>
      <td>0.272762</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.140512</td>
      <td>0.259598</td>
      <td>0.055556</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.135396</td>
      <td>0.253078</td>
      <td>0.063492</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.123114</td>
      <td>0.268165</td>
      <td>0.047619</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.110811</td>
      <td>0.273936</td>
      <td>0.055556</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.111876</td>
      <td>0.260363</td>
      <td>0.047619</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.107146</td>
      <td>0.252084</td>
      <td>0.047619</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


Based on these results we can see that our model is kind of overfitting and we used excess number of epochs in the model above. We can tell that by seeeing that the valid_loss kind of bounces around as does the error_rate too. I think we are comfortable with our `stage-3` model here, so let's do some interpretation using that one:

```python
learn_cleaned_set.load('stage-3');
```

```python
interp = ClassificationInterpretation.from_learner(learn_cleaned_set)
```

```python
interp.plot_confusion_matrix()
```


![png](Passer-classifier_files/output_83_0.png)


```python
learn_cleaned_set.show_results()
```


![png](Passer-classifier_files/output_84_0.png)


Most of our predictions hit the spot here, but no model is perfect as they should not be. We can still see that our dataset isn't perfect as there is still at least one female picture in these pictures. This is the side effect when loading the images from Google Image search as it often contains wrong pictures or some mislabeled ones.

But overall, this was quite fast process to test out some CNN classifiers with my own created dataset. In the futrue I might do another CNN classifier, this time using labels from more than two classes (multi-class). Feel free to try this out and have fun while doing it!
