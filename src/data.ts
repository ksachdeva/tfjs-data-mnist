import * as tf from '@tensorflow/tfjs-core';
import * as tfd from '@tensorflow/tfjs-data';
import {iteratorFromItems, LazyIterator} from '@tensorflow/tfjs-data/dist/iterators/lazy_iterator';

const IMAGE_H = 28;
const IMAGE_W = 28;

const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

class ImageDataset extends tfd.Dataset<tfd.DataElement> {
  private _data: Array<[tf.Tensor1D, tf.Tensor1D]> = new Array();

  public constructor(images: Float32Array, labels: Uint8Array) {
    super();

    const numberOfSamples = images.length / IMAGE_SIZE;

    let imageIndex = 0;
    let labelIndex = 0;
    for (let i = 0; i < numberOfSamples; i++) {
      const imageTensor =
          tf.tensor1d(images.slice(imageIndex, imageIndex + IMAGE_SIZE));
      const labelTensor =
          tf.tensor1d(labels.slice(labelIndex, labelIndex + NUM_CLASSES));
      imageIndex += IMAGE_SIZE;
      labelIndex += NUM_CLASSES;

      this._data.push([imageTensor, labelTensor]);
    }
  }

  async iterator(): Promise<LazyIterator<tfd.DataElement>> {
    return iteratorFromItems<tfd.DataElement>(this._data);
  }
}

export class MNISTDataset {
  trainDataset: tfd.Dataset<tfd.DataElement> = null;
  testDataset: tfd.Dataset<tfd.DataElement> = null;

  private constructor() {}

  static async create() {
    const result = new MNISTDataset();
    await result.loadData();
    return result;
  }

  private async loadData() {
    console.log('* Downloading data .. *');

    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    let datasetImages: Float32Array = null;
    let datasetLabels: Uint8Array = null;

    const imgRequest = new Promise((resolve, _reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }

        datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

    datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // TRAIN
    const trainImages = datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    const trainLabels =
        datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    // TEST
    const testImages = datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    const testLabels = datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    this.trainDataset = new ImageDataset(trainImages, trainLabels);
    this.testDataset = new ImageDataset(testImages, testLabels);
  }
}