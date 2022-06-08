
## Make your data have a format that is supported on the platform

When creating a dataset zip you can load the data in two separate ways.
Both require loading the data in a Pandas `DataFrame` and assume all columns
only contain one type of data with the same shape.

### 1) Load data in memory objects

Store objects directly in the `Series` (columns of your `DataFrame`). This
works for all scalars (floats, integers and strings of one dimension) as well
as [Pillow](https://pillow.readthedocs.io/en/stable/) images and numpy arrays.

#### Example

This is such an example with a progressbar enabled:

```python
df.head()
```

```text
float_column          image_column                numpy_column
    0.248851  <PIL.Image.Image ...  [0.18680, 0.61951, 0.83...
    0.523621  <PIL.Image.Image ...  [0.75213, 0.44948, 0.82...
    0.647844  <PIL.Image.Image ...  [0.41525, 0.63858, 0.34...
    0.447717  <PIL.Image.Image ...  [0.79373, 0.24514, 0.94...
    0.194222  <PIL.Image.Image ...  [0.12636, 0.40554, 0.66...
```

```python
import sidekick

# Create dataset
sidekick.create_dataset(
    'path/to/dataset.zip',
    df,
    progress=True
)
```

### 2) Load data in paths to objects

Columns may also point to paths of object. Which columns are paths should be
indicated in the `path_columns`. Like the in-memory version these may also be
preprocessed.

#### Example

This is an example where all images are loaded from a path,
preprocessed to have the same shape and type and then placed in the dataset.

```python
df.head()
```

```text
float_column string_column                                  image_file_column
    0.248851           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.523621           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.647844           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.447717           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.194222           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
```

```python
import functools
import sidekick

# Create preprocessor for images, cropping to 32x32 and formatting as png
image_processor = functools.partial(
    sidekick.process_image, mode='center_crop_or_pad', size=(32, 32), file_format='png')

# Create dataset
sidekick.create_dataset(
    'path/to/dataset.zip',
    df,
    path_columns=['image_file_column'],
    preprocess={
        'image_file_column': image_processor
    }
)
```
