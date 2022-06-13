
## Upload dataset through the Peltarion Platform's Data API

Peltarion provides a public Data API that enables the users to programmatically get data into the
platform.


#### Example

This example shows how to upload a single file to the Peltarion Platform.

![dataAPI example](../static/image/dataAPI_example.png "Data API example")

Use the `url` and `token` displayed in the modal that appears when clicking the `Data API` button in the Dataset view.

```python
import sidekick

client = sidekick.DatasetClient(url='<url>', token='<token>')
```

This dataset client may now be used to upload one or many files to the Data API service. Uploading files will create a
dataset to the project that the token is tied to. The Data API consumer could provide `dataset_name` and
`dataset_description` to the dataset. If omitted, default name and description will be set to `Sidekick upload`

```python
dataset_name='My dataset'
dataset_description='My description'
filepaths = ['path/to/dataset.zip']

response = client.upload_data(
    filepaths=filepaths,
    name=dataset_name,
    description=dataset_description,
)
```

![dataset_upload example](../static/image/dataset_upload_example.png "Dataset upload example")
