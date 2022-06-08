
## Make predictions with a trained and deployed model on the Platform

To connect to an enabled deployment use the `sidekick.Deployment` class. This
class takes the information you find on the deployment page of an experiment.

#### Example

This example shows how to query an enabled deployment for image classification.

![deployment example](static/image/deployment_example.png "Deployment example")

Use the `url` and `token` displayed in the dark box.

```python
import sidekick

client = sidekick.Deployment(url='<url>', token='<token>')
```

This deployment client may now be used to get predictions for images.

The feature specifications from the table of input and output parameters can be accessed as a
property of the client object:

```python
# input features
client.feature_specs_in

# output features
client.feature_specs_out
```

### Test deployment with one sample - predict

To predict result of one image (here `test.png`) use `predict`.

#### Example

```python
from PIL import Image

# Load image
image = Image.open('test.png')

# Get predictions from model
client.predict(image=image)
```

Note: If the feature name is not a valid python variable, e.g., `Image.Input`, use `predict_many` instead of `predict`.

### Test deployment with many samples - predict_many

To efficiently predict the results of multiple input samples (here, `test1.png`, `test2.png`) use
`predict_many`.

#### Example

```python
client.predict_many([
    {'image': Image.open('test1.png')},
    {'image': Image.open('test2.png')}
])
```

### Interactive exploration of data - predict_lazy

For interactive exploration of data it is useful to use the `predict_lazy`
method, which returns a generator that lazily polls the deployment when needed.
This allows you to immediatly start exploring the results instead of waiting
for all predictions to finnish.

#### Example

```python
client.predict_lazy([
    {'image': Image.open('test1.png')},
    {'image': Image.open('test2.png')}
])
```

### Compatible filetypes

The filetypes compatible with sidekick may shown by:

```python
print(sidekick.encode.FILE_EXTENSION_ENCODERS)
```
