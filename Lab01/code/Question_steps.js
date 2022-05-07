// --------------------------------------------------
// REPORT'S QUESTIONS - ZITOUMENA
// --------------------------------------------------

// --------------------------------------------------
// QUESTION 2
// Για την περιοχή μελέτης παρουσιάστε ένα φυσικό και
// ένα ψευδέχρωμο σύνθετο καθώς και το δείκτη NDVI για την 
// ημερομηνία με τα λιγότερα σύννεφα κατά το 2019.
// --------------------------------------------------

// already implemented at the beginning of code


// --------------------------------------------------
// QUESTION 3
// Παρουσιάστε και περιγράψτε το χάρτη με τις μέγιστες για το 2019 
// τιμές NDVI και τον χάρτη με τις αντίστοιχες ημερομηνίες (DOY) για
// τις οποίες είχαμε αυτές τις τιμές. Πραγματοποιήστε το ίδιο για το
// 2018 και συγκρίνετε τα αποτελέσματα. Που οφείλονται οι διαφορές?
// --------------------------------------------------

var l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1"); //dataset import
var images_2019 = l8.filterDate("2019-01-01", "2019-12-31")
                .filterBounds(polygon); 
var images_2018 = l8.filterDate("2018-01-01", "2018-12-31")
                .filterBounds(polygon);  

//Map.centerObject(polygon, 13); //center scene to polygon, 13 is the zoom level here


//the following two layers are used to clip/isolate cut a rectangular from the above layers
Map.centerObject(rect, 10);

// Compute NDVI for image timeseries
var addNDVI = function(image) {
  // NDVI = (NIR - Red) / (NIR + Red)
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};


var ndvi_images_2019 = images_2019.map(addNDVI); //apply the function to ImageCollection, returns ImageCollection
var mosaic_image_2019 = ndvi_images_2019.qualityMosaic('NDVI');

var ndvi_images_2018 = images_2018.map(addNDVI); //apply the function to ImageCollection, returns ImageCollection
var mosaic_image_2018 = ndvi_images_2018.qualityMosaic('NDVI');


// RYG, red yellow green pallette with red = least vegetation, green = max vegetation
var ndvispectrum = {'min': 0, 'max': 0.5, 'palette': ['orangered','yellow','green'], bands: 'NDVI'};
Map.addLayer(mosaic_image_2019, ndvispectrum, '2019 Mosaic image with max NDVI pixel values - NDVI band');
Map.addLayer(mosaic_image_2019.clip(rect), ndvispectrum, '2019 Mosaic image with max NDVI pixel values - NDVI band - rect');

Map.addLayer(mosaic_image_2018, ndvispectrum, '2018 Mosaic image with max NDVI pixel values - NDVI band');
Map.addLayer(mosaic_image_2018.clip(rect), ndvispectrum, '2018 Mosaic image with max NDVI pixel values - NDVI band - rect');


var addDoy = function(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('DOY');
  doyBand = doyBand.updateMask(image.select('NDVI').mask());
  return image.addBands(doyBand);
};

// add DOY (day of year) as a band in the images of the image collection
var images_2bands_extra_2019 = ndvi_images_2019.map(addDoy); // add the extra band DOY
var mosaic_image_with_doy_2019 = images_2bands_extra_2019.qualityMosaic('NDVI');

var images_2bands_extra_2018 = ndvi_images_2018.map(addDoy); // add the extra band DOY
var mosaic_image_with_doy_2018 = images_2bands_extra_2018.qualityMosaic('NDVI');

var max_ndvi_doy_spectrum = {'min': 1, 'max': 366, 
  'palette': ['yellow', 'lightgreen', 'green', 'red'], bands: 'DOY'};

Map.addLayer(mosaic_image_with_doy_2019, max_ndvi_doy_spectrum, '2019 DOY (day out of 365) with the max NDVI per pixel');
Map.addLayer(mosaic_image_with_doy_2019.clip(rect), max_ndvi_doy_spectrum, '2019 DOY (day out of 365) with the max NDVI per pixel - rect');

Map.addLayer(mosaic_image_with_doy_2018, max_ndvi_doy_spectrum, '2018 DOY (day out of 365) with the max NDVI per pixel');
Map.addLayer(mosaic_image_with_doy_2018.clip(rect), max_ndvi_doy_spectrum, '2018 DOY (day out of 365) with the max NDVI per pixel - rect');


// --------------------------------------------------
// QUESTION 4
// Για 4 διαφορετικά πολύγωνα στην περιοχή μελέτης, για όλα τα διαθέσιμα 
// δεδομένα L8 και τις χρονοσειρές NDVI (ή άλλων δεικτών για τον αστικό χώρο,
// τα ύδατα, το έδαφος), περιγράψτε και παρουσιάστε τις βέλτιστες αρμονικές
// καμπύλες που προσαρμόσατε δημιουργώντας συνθετικές χρονοσειρές.
// --------------------------------------------------

var images_all = l8.filterBounds(polygon).filter(ee.Filter.lt('CLOUD_COVER', 20)); //.sort("CLOUD_COVER");
var ndvi_images_all = images_all.map(addNDVI); //add NDVI band



// HARMONIC LINE model fit to data
// This field contains UNIX time in milliseconds.
var timeField = 'system:time_start';

// Use this function to add variables for NDVI, time and a constant
// to Landsat 8 imagery.
var addVariables = function(image) {
  // Compute time in fractional years since the epoch.
  var date = ee.Date(image.get(timeField));
  var years = date.difference(ee.Date('1970-01-01'), 'year');
  // Return the image with the added bands.
  return image
    // Add a time band.
    .addBands(ee.Image(years).rename('t').float())
    // Add a constant band.
    .addBands(ee.Image.constant(1));
};

var ndvi_images_all = ndvi_images_all.map(addVariables);



// Harmonic trend ----------------------------------------------------------------
// Use these independent variables in the harmonic regression.
var harmonicIndependents = ee.List(['constant', 't', 'cos', 'sin']);

// Add harmonic terms as new image bands.
var harmonicLandsat = ndvi_images_all.map(function(image) {
  var timeRadians = image.select('t').multiply(2 * Math.PI);
  return image
    .addBands(timeRadians.cos().rename('cos'))
    .addBands(timeRadians.sin().rename('sin'));
});
  
// The output of the regression reduction is a 4x1 array image.
var harmonicTrend = harmonicLandsat
  .select(harmonicIndependents.add('NDVI'))
  .reduce(ee.Reducer.linearRegression(harmonicIndependents.length(), 1));

// Turn the array image into a multi-band image of coefficients.
var harmonicTrendCoefficients = harmonicTrend.select('coefficients')
  .arrayProject([0])
  .arrayFlatten([harmonicIndependents]);

// Compute fitted values.
var fittedHarmonic = harmonicLandsat.map(function(image) {
  return image.addBands(
    image.select(harmonicIndependents)
      .multiply(harmonicTrendCoefficients)
      .reduce('sum')
      .rename('fitted'));
});

// Plot the fitted model and the original data at the ROI.
print(ui.Chart.image.series(
  fittedHarmonic.select(['fitted','NDVI']), polygon, ee.Reducer.mean(), 30)
    .setSeriesNames(['NDVI', 'fitted'])
    .setOptions({
      title: 'Harmonic model: original and fitted values - polygon 1',
      lineWidth: 1,
      pointSize: 3,
}));


// Plot the fitted model and the original data at the ROI.
print(ui.Chart.image.series(
  fittedHarmonic.select(['fitted','NDVI']), polygon2, ee.Reducer.mean(), 30)
    .setSeriesNames(['NDVI', 'fitted'])
    .setOptions({
      title: 'Harmonic model: original and fitted values - polygon 2',
      lineWidth: 1,
      pointSize: 3,
}));


// Plot the fitted model and the original data at the ROI.
print(ui.Chart.image.series(
  fittedHarmonic.select(['fitted','NDVI']), polygon3, ee.Reducer.mean(), 30)
    .setSeriesNames(['NDVI', 'fitted'])
    .setOptions({
      title: 'Harmonic model: original and fitted values - polygon 3',
      lineWidth: 1,
      pointSize: 3,
}));


// Plot the fitted model and the original data at the ROI.
print(ui.Chart.image.series(
  fittedHarmonic.select(['fitted','NDVI']), polygon4, ee.Reducer.mean(), 30)
    .setSeriesNames(['NDVI', 'fitted'])
    .setOptions({
      title: 'Harmonic model: original and fitted values - polygon 4',
      lineWidth: 1,
      pointSize: 3,
}));



// --------------------------------------------------
// QUESTION 5
// Για την ίδια ή διαφορετική περιοχή μελέτης, ψηφιοποιήστε διάφορες χρήσεις γης
// και πραγματοποιείστε ταξινόμηση στην διαθέσιμη χρονοσειρά. Χρησιμοποιείστε δυο
// διαφορετικούς ταξινομητές. Περιγράψτε τυχόν διαφορές που προκύπτουν στα 
// αποτελέσματα της ταξινόμησης σε σχέση με τις διαθέσιμες ψηφιοποιημένες 
// περιοχές ή τους διαφορετικούς ταξινομητές.
// --------------------------------------------------

// Step 2.8: Classification with CART
//Training data creation
var label = 'Class';
var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
              'B7', 'B8', 'B9', 'B10', 'B11', ];
              
var input = images_all.sort("CLOUD_COVER").first().select(bands); //select image with min cc and bands

var training = Urban.merge(Water).merge(Vegetation); //.merge(Streets);
print('Training marker points:', training);


//overlay the points on the image to get training
var trainImage = input.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});

print('Training image:', trainImage)

var trainingData = trainImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lessThan('random', 0.8));
var testSet = trainingData.filter(ee.Filter.greaterThanOrEquals('random', 0.8));

//classification models
var CART_classifier = ee.Classifier.smileCart().train(trainSet, label, bands);
var SVM_classifier =  ee.Classifier.libsvm({kernelType: 'LINEAR', cost: 1}).train(
    trainSet, label, bands);

//classify the image
var classified_CART = input.classify(CART_classifier);
var classified_SVM = input.classify(SVM_classifier);

print('Classified image CART:', classified_CART.getInfo());
print('Classified image SVM:', classified_SVM.getInfo());

var landcoverPalette = [
  '#0c2c84', //water (class 0)
  '#e31a1c', //urban (class 1)
  '#005a32', //vegetation (class 2)
  '#808080', //streets (class 3)
  ];
  
Map.addLayer(classified_CART, {palette: landcoverPalette, min:0, max:3}, 'classification - CART');
Map.addLayer(classified_CART.clip(rect), {palette: landcoverPalette, min:0, max:3}, 'classification - CART - rect');

Map.addLayer(classified_SVM, {palette: landcoverPalette, min:0, max:3}, 'classification - SVM');
Map.addLayer(classified_SVM.clip(rect), {palette: landcoverPalette, min:0, max:3}, 'classification - SVM - rect');


//accuracy assessment
var confusionMatrix_CART = ee.ConfusionMatrix(testSet.classify(CART_classifier)
  .errorMatrix({
    actual: 'Class',
    predicted: 'classification'
  }));


var confusionMatrix_SVM = ee.ConfusionMatrix(testSet.classify(SVM_classifier)
  .errorMatrix({
    actual: 'Class',
    predicted: 'classification'
  }));

  
  
print('Confusion Matrix CART:', confusionMatrix_CART);
print('Overall Accuracy CART:', confusionMatrix_CART.accuracy());

print('Confusion Matrix SVM:', confusionMatrix_SVM);
print('Overall Accuracy SVM:', confusionMatrix_SVM.accuracy());


