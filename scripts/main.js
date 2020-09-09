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

async function plot(pointsArray, featureName) {
    tfvis.render.scatterplot(
        {
            name: `${featureName} vs House Price` 
        },
        {
            values: [pointsArray], series: ["original"]
        },
        {
            xLabel: featureName,
            yLabel: "Price"
        }
    )
}

function normalize(tensor) {
    const min = tensor.min()
    const max = tensor.max()
    const normalizedTensor = tensor.sub(min).div(max.sub(min))
    return {
        tensor: normalizedTensor,
        min,
        max
    }
}

function denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min)
    return denormalizedTensor
}

async function run () {
    const houseSalesDataSet = tf.data.csv("http://127.0.0.1:5500//kc_house_data.csv")
    // const sampleDataSet = houseSalesDataSet.take(10)
    // const dataArray = await sampleDataSet.toArray()
    // console.log(dataArray)

    const points = houseSalesDataSet.map(record => ({
        x: record.sqft_living,
        y: record.price
    }))
    plot(await points.toArray(), "Square feet")

    // features (inputs) 
    const featureValues = await points.map(p => p.x).toArray()
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])

    //labels (outputs)
    const labelValues = await points.map(p => p.y).toArray()
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    featureTensor.print()
    labelTensor.print()

    // const normalizedFeature = normalize(featureTensor)
    // const normalizedLabel = normalize(labelTensor)

    // normalizedFeature.tensor.print()
    // normalizedLabel.tensor.print()

    // denormalize(normalizedFeature.tensor, normalizedFeature.min, normalizedFeature.max).print()

}

run()
  