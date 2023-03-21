# ddim-cars

image generator which generate cars images!


## usage
```python
from main import Diffusion
d = Diffusion()
d.plot_images()
```
the output image will be saved in the `new_predictions` folder


## training:
first run the `process_data()` function
then call the `train()` function.
```python
from main import Diffusion
d = Diffusion()
d.train()
```
## example:
<p align="left">
  <img width="500" src="https://github.com/matan-chan/ddim-cars/blob/main/output_images/generated_plot_epoch-50250.png?raw=true">
</p>

## data:
[Cars Dataset][website]



[website]: http://ai.stanford.edu/~jkrause/cars/car_dataset.html