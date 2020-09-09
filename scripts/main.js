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

async function run () {
    const houseSalesDataSet = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv")
    const sampleDataSet = houseSalesDataSet.take(10)
    const dataArray = await sampleDataSet.toArray()
    console.log(dataArray)

    const points = houseSalesDataSet.map(record => ({
        x: record.sqft_living,
        y: record.price
}))
plot(await points.toArray(), "Square feet")
}

run()
  