import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// installing, section 2
// // Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// // Train the model using the data.
// model.fit(xs, ys, {epochs: 10}).then(() => {
//   // Use the model to do inference on a data point the model hasn't seen before:
//   model.predict(tf.tensor2d([5], [1, 1])).print();
//   // Open the browser devtools to see the output
// });
// console.log(tf.getBackend());

// section 3
// tf.tidy(() => {const xs = tf.tensor1d([1, 2, 3]);
// const ys = xs.add(tf.tensor2d([[5], [6], [7]]));
// ys.print();})
// function getYs(xs, m, c) {
//   // To implement
//   return xs.mul(m).add(c);
// }
// tf.tidy(() => {const t1 = tf.tensor1d([1,5,10]);
// const t2 = getYs(t1, 2, 1);
// t2.print();});
// tf.tidy(() => {const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22]);
// const max = t3.max(); // 76
// const min = t3.min(); // -5
// const range = max.sub(min);
// t3.sub(min).div(range).print();})
// for (let i = 0; i < 100; i++) {
//   const tensor1 = tf.tensor1d([1,2,3]);
//   tf.dispose(tensor1);
// }
// tf.tidy(() => {
//   for (let i = 0; i < 100; i++) {
//   tf.tensor1d([4,5,6]).print();
// }});
// console.log(tf.memory());

function plot(points, featureName) {
  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values: [points], series: ['original'] },
    {
      xLabel: featureName,
      yLabel: 'Price'
    }
  );
}

function normalize (tensor) {
  const max = tensor.max();
  const min = tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));
  return { tensor: normalisedTensor, min, max };
}

function denormalise (tensor, min, max) {
  return tensor.mul(max.sub(min)).add(min);
}

async function run() {
  await tf.ready();
  const houseSalesData = tf.data.csv('/kc_house_data.csv');

  const pointsDataset = houseSalesData.map(record => ({
    x: record.sqft_living, y: record.price
  }));
  const points = await pointsDataset.toArray();
  tf.util.shuffle(points);
  if (points % 2 !== 0) points.pop();

  tf.tidy(() => {
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);
    const normalizedFeature = normalize(featureTensor);
    const normalizedLabel = normalize(labelTensor);

    const [ trainingFeatures, testingFeatures ] = tf.split(normalizedFeature.tensor, 2);
    const [ trainingLabels, testingLabels ] = tf.split(normalizedLabel.tensor, 2);
    // tf.keep(trainingFeatures);
  });
}

run().then(() => console.log(tf.memory()));