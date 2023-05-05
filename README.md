Ufilm Segment
=============

**License:** Apache-2.0

## Introduction

This is a proof of concept for [Internet Archive](https://archive.org) of text segmentation/extraction for microfilm archives. 
Like many libraries, the Archive has thousands of journals and magazines on microfilm which has been digitized. 
The issues are organized only as images plus OCR hocr files with some metadata. 
What is missing is a clickable table of contents to show specific articles. 
This project was able to show that a U-shaped deep learning neural network is able to be trained to extract article 
titles, authors and bibliography to make a table of contents. Accuracy was around 95% for single journal training.

The U-shaped neural network is from the [dhSegment](https://github.com/tralfamadude/dhSegment) project which is derived 
from the wonderful work done by the Digital Humanities Laboratory 
[dhSegment](https://github.com/dhlab-epfl/dhSegment) project at [EPFL](http://dhlab.epfl.ch).

## Image Annotation
For image annotation, CVAT is used from the OpenVINO project from Intel. There is no direct linkage with CVAT to this code; 
CVAT should be run in a docker with a shared directory accessible by the docker instance and other host processes. 
A python script (fetch_annotations.py) is used to fetch the annotations via CVAT API, 
so there is no particular annotation format in use. 

![Annotation Process Diagram](diagrams/Annotation_Process.png)

For semantic segmenation, a U-shaped network maps image pixels (the X of the neural network) to mask pixels (the Y). 
The mask and image are the same size and colors represent the segmentation class against a black background. 
A csv file is used to name the classes (config/annotation_names.csv) which correspond to colors from a predefined list
of available colors (in annotation_convert.py).  

## Training Process
The images from the microfilm journal are paired with annotated image masks as training data. 
They are split into training, evaluations, and test sets which are used to train the dhSegment model. 
Then a secondary classic ML model (the "post model", a decision tree) is trained to take the output of the dhSegment 
model and predict/classify which kind of page it is. 
For example, the page might be the table of contents or article start or bibliography. 
This page classifier directs postprocessing to take appropriate actions.

![Training Process Diagram](diagrams/Training_Process.png)

## Distributed Computing
If we naively just wrote one simple process with one thread that fetched an image, did inference with the GPU, and then did 
postprocessing, the GPU utilization would be very low, like less than 20%. 
That's because the GPU is very fast compared to the steps in the CPU. 
To improve GPU utilization, this project uses (Ray)[https://github.com/ray-project/ray] to 
increase the number of CPU cores used so that the GPU can be fully utilized when processing a stream of images from 
microfilm journal issues.

Python has a Global Interpreter Lock (GIL) that protects data access that would get corrupted
if it were accessed by multiple threads (cores); the GIL has been in Python since the beginning and 
it is very hard to eliminate by splitting it into multiple locks and coordinated all the code and libraries
to use it properly, so it is unlikely to go away soon. 
Multiple threading and process packages attempt to remedy this liveliness challenge, but they 
amount to co-routines because only one can run at a time. 

Ray enables parallelism by creating full OS processes for each agent or worker. 
A full OS process will clearly not be blocked by the Python GIL in another process.
Ray also provides queues and message passing. 
The queues are used for flow control between steps. 
Putting something on a queue generally does not block, but we can put a bound on a queue length to have a kind
of forward flow control so that if early stages are always operating faster than later ones, the early processing 
can pause to let the later stages catch up. 
Bounded queues also can prevent a crashed later stage from causing a queue to grow indefinitely.  
Popping something off a queue will block so that consumers do not need to poll. 

One journal issue is processed at a time as an ordered list of images, one image per page. 
The images are fetched from disk and fed to the U-shaped neural network on the GPU without any preprocessing; the microfilm images are 
already de-skewed and normalized. (The de-skewing had to be done before OCR to eliminate any accidental 
rotation caused by improper alignment when the microfilm images were captured.)

The result of the neural network inference step is a set of features (features.py) which are put into a queue and 
multiple instances of the page post-processing agents/workers pop inferred features off the queue to do page level processing. 
 
Page level processing uses the "post model", a decision tree, which takes the features and decides
what to do next. 
For example, if the page is the start of an article because it has a block of text classified as a 
title, then article-start actions are performed. 
After page-level processing is done, the results are put on another queue to wait for issue-level processing, 
called the finisher. 

The finisher post-processing does not commence until all the images from a journal issue are processed by the GPU. 
This enables the postprocessing to see all the inferences for the issue at once and that means it is able 
to analyze the content in a holistic way. 
Knowing when all the pages have been processed is a little tricky. 
Since the goal is to stream multiple issues through the pipeline, it is possible that the page-level post-processing 
for one issue could be happening at the same time as another issue starts. 
To help the finisher agent to recognize this, metadata about the issue can be pushed on to 
the second queue when an issue begins or ends. 
This notifies the finisher that all the pages will be ready soon. 
The finisher knows how many pages there are and can check to see when the issue is complete. 
If a page from another issue is found, it can be put back on the queue. 

As an example of finisher work, the bibliography content can be associated with the article start page 
that came before it; 
while this is not needed to construct a table of contents, it turns out that identifying bibliographic
references during semantic segmentation is not that difficult because references look like 
many small paragraphs and in bulk it has a certain texture that is easy for a convolutional model to spot. 

The bibliography is useful for creating a citation graph. 
Extracting references was a nice-to-have goal; it came along for the ride without much effort. 


![Segmentation Dataflow](diagrams/Microfilm_Segmentation_Dataflow.png)



## Results
The output is a json that describes the articles found in a journal issue. 
The article title, authors, page number, are included. 

## Conclusion
The tests on one journal showed that accuracy of extraction reached 95% in the test set. 
More accuracy, generalization, and throughput tests would be helpful. 

There are thousands of journals in the collection, so next step would be to bulk-annotate 
perhaps a hundred journal examples and then retrain and test for accuracy on journals that 
were not in the training set. 
Currently, there is no funding for this so the project remains a proof of concept. 

My favorite part of this project is the ML Ops part where Ray is used for parallelism. 
Using Ray to increase CPU core utilization can be a good technique, although the ratio of message
processing to message sending time needs to be sufficiently high for it to be efficient. 
I focused on the Ray design patterns used here which can be helpful to others as an example of 
using Ray to process parts (page-level) and aggregations (issue-level) in a pipeline. 

## Public Note
The microfilm image collection at Internet Archive is not directly accessible by the public
because most journals insist that only one person at a time can borrow it to read. 
The public can borrow an issue at a time and read it online, but the bulk images cannot be downloaded. 
That means we cannot include the dataset here. 
If you want to try this out, you will need to supply a dataset. 
