# Our Kaggle Commonlit solution - Story of how we climbed up to 40th position from 1100th position in 3 days
## About our team - Southies
### Vignesh Baskaran
### Shahul ES

## Competition Objective
The objective of this competition is to assess the readability of a text snippet. 


## Challenges: 
Although Shahul had much experience with transformers earlier, this was my first experience. Therefore I was very surprised with several of the challenges transformers posed and it was very adventurous to discover tricks to tackle these challenges.

### Challenge 1: Reproducibility
In the begining of the competition it was very difficult for us to reproduce our own results. We had to carefully seed_everything to exactly reproduce our own training scripts. Here is the final `seed_everything` function we used:

 
```python

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

```

### Challenge 2: Size of the dataset
One of the main challenges of the training dataset is that the labels were quite **noisy** and also the size of the dataset was quite **small**. The training dataset had only `2834` samples. Therefore since the very begining we were quite sure that we need some external data to pretrain/train the model. 

```raw
    Length of train data: 2267
    Length of valid data: 567
    Number of batches in the train_dataloader: 284
    Number of batches in the valid_dataloader: 71
```

### Challenge 3: Training with external dataset
The competition labels were not intuitive. Basically every text snippet is assigned a score (a scalar value) based on how difficult it is to read. Essentially if all the text snippets were sorted based on the labels they would be sorted to reflect their difficulty. Although some external datasets were available with similar kind of data it was not possible to utilize them for this competition  either because they were proprietary or very small. Therefore we tried to understand how the target labels were generated in the training dataset. The target scores were not assigned by the raters directly because the targets are not at all intuitive. Instead pairs of text snippets were compared by examiners and they rated which text is easy to read and which text is difficult to read in the pair. We wanted to simulate the same process to harvest large amount of external data. We initially thought we would label pairs of texts using Mechanical turk but we ruled out that option as were not sure if that was acceptable. Then we got another idea that was extremely efficient. We found that Wikipedia has a counterpart called Simple Wikipedia which essentially contains several of the Wikipedia articles written in Simple English. For instance here is the article on `Rocket` from `Wikipedia`:


Wikipedia | Simple Wikipedia
:---------------:|:---------------:
A rocket is a projectile that spacecraft, aircraft or other vehicles use to obtain thrust from a rocket engine. Rocket engine exhaust is formed entirely from propellant carried within the rocket. Rocket engines work by action and reaction and push rockets forward simply by expelling their exhaust in the opposite direction at high speed, and can therefore work in the vacuum of space. [Link](https://wikipedia.org/wiki/Rocket)| A rocket may be a missile, spacecraft, aircraft or other vehicle which is pushed by a rocket engine. Some big rockets are launch vehicles and some are manned . Other rockets, for example missiles, are unmanned. ("Manned" means that a person is in it; "unmanned" means that the machine goes without a person.) Most rockets can be launched from the ground because exhaust thrust from the engine is bigger than the weight of the vehicle on Earth. [Link](https://simple.wikipedia.org/wiki/Rocket)

This looked fantastic to us. We also found a dataset which was scraped and preprocessed by available [here](https://www.kaggle.com/markwijkhuizen/simplenormal-wikipedia-abstracts-v1). We are very grateful to Mark Wijkhuizen for creating this fantastic dataset. Therefore we decided to make sure that we write our code that is generic enough to train both the external dataset and internal dataset. 

### Challenge 4: Robust code
Writing code that is generic enough to train external as well as competition data was quite challenging. We wanted to run very different experiments but we also wanted to make sure that they are run in the same setup so that we can compare them easily. For instance we had the following options for the source of the data:
 1. Pretrain a ranker on external data and then finetune a regressor on competition data
 2. Just train a regressor on Competition data
 3. MLM pretrain and then finetune a regressor on competition data
 
 We also had the following options of models:
 1. Roberta-large
 2. Roberta-base
 
 And then we had three different learning rate assigning strategies to test
 1. Uniform learning rate
 2. Layer-wise learning rate decay
 3. Grouped learning rate
 
 In order to facilitate this extensive experimentation we created a very robust code and it took us more than a week and over 70 commits to make it stable.
 
 ### Challenge 5: Unstable training
 One of the biggest challenges in training the algorithm was the unstability that the algorithm exposed. 
 

