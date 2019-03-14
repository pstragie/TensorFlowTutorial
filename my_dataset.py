import tensorflow_datasets as tfds

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("This is the dataset for histopathology. The "
                   "images are kept at their original dimensions."),
      # tfds.features.FeatureConnectors
      features=tfds.features.FeaturesDict({
        "image_description": tfds.features.Text(),
        "image": tfds.features.Image(),
        # Here, labels can be of 5 distinct values.
        "label": tfds.features.ClassLabel(num_classes=2),
      }),
      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=("image", "label"),
      # Homepage of the dataset for documentation
      urls=["https://www.kaggle.com/c/histopathologic-cancer-detection/data"],
      # Bibtex citation for the dataset
      citation=r"""@article{dataset-from-kaggle,
                                  author = {kaggle, Competition},"}""",
    )

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    pass  # TODO

  def _generate_examples(self):
    # Yields examples from the dataset
    pass  # TODO