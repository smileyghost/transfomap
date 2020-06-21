# Transformap: Predicting the time arrival with Transformer

## Dataset
We consider three components that are useful for the time estimation:
- Distance
- Rain Fall Rate
- Trajectories in map

First, we collect 28000 map images based on mapmatched GPS trajectories for each transaction. 
Then, we use [OSRM](http://project-osrm.org/) hosted in local machine to perform mapmatching due to its robustness and scalablility.
We render the map images with [Azure Map Render service](https://docs.microsoft.com/en-us/rest/api/maps/render/getmapimage) that shows the uploaded GPS trajectories. Finally, we split the dataset as train, valid, test subset with 8:1:1.

## Model
Our model consist of Convolution Neural Network (ConvNet) and Self-Attention Transformer. Inspired by [DETR](https://github.com/facebookresearch/detr), we use Convolution Layer to extract features from the map image then we feed the to the encoder of Transformer. Then, we feed the decoder with the query embeddings that consists of:

- rainfall_rate * rain_fall embedding
- distance * distance embedding
- time_of_day (0000-2359, in 24 hour formats, e.g., 2359)
- day_of_week (0-6, 0 denoted as monday)

## Model Serving
We deploy our model in AzureML to track our updated model. We serve the model with AzureML Webservice. To handle the traffics, we put [frontend_apps.py](frontend_apps.py) in front of the webservice.
