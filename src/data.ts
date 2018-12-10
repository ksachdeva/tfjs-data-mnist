import * as tf from '@tensorflow/tfjs';
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

class ImageDataset extends tf.data.Dataset<tf.data.DataElement> {
  private _data: Array<[tf.Tensor1D, tf.Tensor1D]> = new Array();
  private array = new Array();

  public constructor(images: Float32Array, labels: Uint8Array) {
    super();

    const numberOfSamples = images.length / IMAGE_SIZE;

    let imageIndex = 0;
    let labelIndex = 0;
    for (let i = 0; i < 12 /* numberOfSamples*/; i++) {
        const imageArray = images.slice(imageIndex, imageIndex + IMAGE_SIZE);
      const imageTensor =
          tf.tensor1d(imageArray);
          const labelArray =labels.slice(labelIndex, labelIndex + NUM_CLASSES);
      const labelTensor =
          tf.tensor1d(labelArray);
      imageIndex += IMAGE_SIZE;
      labelIndex += NUM_CLASSES;

      this._data.push([imageTensor, labelTensor]);
      this.array.push([imageArray, labelArray]);
    }
  }

  async iterator(): Promise<LazyIterator<tf.data.DataElement>> {
    return iteratorFromItems<tf.data.DataElement>(this._data);
  }

  getArray() {
      return tf.data.array(this.array);
  }
}

export class MNISTDataset {
  trainDataset: tf.data.Dataset<tf.data.DataElement> = null;
  testDataset: tf.data.Dataset<tf.data.DataElement> = null;

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

    const train = new ImageDataset(trainImages, trainLabels);
    const test = new ImageDataset(testImages, testLabels);

    this.trainDataset = train.getArray();
    this.testDataset = test.getArray();


    // this.trainDataset = new ImageDataset(trainImages, trainLabels);
    // this.testDataset = new ImageDataset(testImages, testLabels);
  }
}
