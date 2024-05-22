Our goal to find out the number of cameras are required in Precision Agriculture where the system detects the number of crops in a given nursery farm, efficiently and even faster. 

The data was collected from differnt nursery farms with different cameras.

The data has been annotated using CVAT and trained with different state of the art object detection models. Then evaluated using Intersection Over Union metric for all the bounding boxes. 
Depending upon the results obtained, found that viewing angles were affecting the IoU scores of these bounding boxes across the frame, and discovered 
the closest viewing angles are posessing the highest mean IoU  and the farther angles were giving the lower mean IoU scores. Finally, we found out that 
YOLOv9c performed relatively well compared to the other detection models.

As our goal to find out the number of cameras are required, in order to get this, we need to track every single crop across different frames using a tracking algorithm
like BoT-SORT, ByteTrack, SORT, DeepSORT, MAATrack.

So, now onto finding the best tracking algorithm that suits well with the data using the object detection model YOLOv9c.

Will update further details.
