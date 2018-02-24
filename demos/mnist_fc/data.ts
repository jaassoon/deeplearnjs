import * as dl from 'deeplearn';

const TRAIN_TEST_RATIO = 5 / 6;

const mnistConfig: dl.XhrDatasetConfig = {
  'data': [
    {
      'name': 'images',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_images.png',
      'dataType': 'png',
      'shape': [28, 28, 1]
    },
    {
      'name': 'labels',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_labels_uint8',
      'dataType': 'uint8',
      'shape': [10]
    }
  ],
  modelConfigs: {}
};

export class MnistData {
  private dataset: dl.XhrDataset;
  trainingData: dl.Tensor[][];

  public async load() {
    this.dataset = new dl.XhrDataset(mnistConfig);
    await this.dataset.fetchData();

    this.dataset.normalizeWithinBounds(0, -1, 1);
    this.trainingData = this.getTrainingData();
  }

  private getTrainingData(): dl.Tensor[][] {
    const [images, labels] =
        this.dataset.getData() as [dl.Tensor[], dl.Tensor[]];

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
  }
}