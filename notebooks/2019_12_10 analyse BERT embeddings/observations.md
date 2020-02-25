- Using tensorboard visualization on 1000 sampled sentences including the word "bank", there are clusters indeed. 
Now it is a question of filtering them out.
- We need to remove all the "hubs". What is a hub defined as, however?
- We don't have the "ego" by using context embeddings. unlike "table" or so. thus, we need to be 
creative by removing hubs within graphs
- Hub detection can also be used when matching one into another, 
which could be super helpful for bilingual token matching

-> Ideas: we can later on create GMMs using the different communities
-> take largest 50 items as hubs..

-> Look up concepts of "community detection" in graph analysis

 