var l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1"); //dataset import

//filter dataset with 2019 images, cloud_cover<20% and sort by cloud_cover ascending
var images = l8.filterDate("2019-01-01", "2019-12-31")
                .filterBounds(polygon)
                .filter(ee.Filter.lt("CLOUD_COVER", 20)); 
  
print("Number of images for 2019 for selected polygon \
with cloud cover less than 20%:", images.size()); 
print("Images for the selected region and daterange and cloud coverage filters:", images);

var image = ee.Image(images.sort("CLOUD_COVER").first()); //get image in 2019 with lowest cloud cover
print("For the image with the lowest cloud coverage out of those:",
  "Image date:", image.get('DATE_ACQUIRED'), "Cloud coverage", image.get('CLOUD_COVER'));
print("Image with minimum cloud coverage:", image);
  
//display image in true color bands (RGB), bands 4, 3 and 2
var RGB432 = {bands: ["B4", "B3", "B2"], min: 5000, max: 12000};
var RGB543 = {bands: ["B5", "B4", "B3"], min: 4000, max: 12000};

//Map.centerObject(polygon, 13); //center scene to polygon, 13 is the zoom level here

//adding layer to the map, using the visualization parameters

Map.addLayer(image, RGB432, "RGB432 - True Color");
Map.addLayer(image, RGB543, "RGB543 - Near InfraRed");

//the following two layers are used to clip/isolate cut a rectangular from the above layers
Map.centerObject(rect, 10);
Map.addLayer(image.clip(rect), RGB432, "RGB432 - True Color rectangular"); 
Map.addLayer(image.clip(rect), RGB543, "RGB543 - Near InfraRed rectangular");

// Step 2.4
// Compute the Normalized Difference Vegetation Index (NDVI) for a single image
var nir = image.select('B5');
var red = image.select('B4');
var ndvi_image = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

var ndviParams = {min: -0.1, max: 0.35, bands:'NDVI', 
    palette: ['orangered','yellow','green'], region: polygon}; //default ndvi values in [-1,1]
Map.addLayer(ndvi_image, ndviParams, 'NDVI image (min cloud_cover)');
Map.addLayer(ndvi_image.clip(rect), ndviParams, 'NDVI image (min cloud_cover) rectangular');

// Step 2.5
// Compute NDVI, EVI, και NDWI for image timeseries

// NDVI
var addNDVI = function(image) {
  // NDVI = (NIR - Red) / (NIR + Red)
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

// EVI
var addEVI = function(image) {
  // EVI = 2.5*(NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
  var nir = image.select('B5');
  var red = image.select('B4');
  var blue = image.select('B2');
  var evi = (nir.multiply(2.5).subtract(red.multiply(2.5)).divide(nir.add(red.multiply(6)).
    add(blue.multiply(-7.5)).add(1))).rename('EVI');
  return image.addBands(evi);
};

// NDWI
var addNDWI = function(image) {
  //For Landsat 8 data, NDWI = (Band 5 – Band 6) / (Band 5 + Band 6)
  var ndwi = image.normalizedDifference(['B6', 'B5']).rename('NDWI');
  return image.addBands(ndwi);
};

var ndvi_images = images.map(addNDVI); //apply the function to ImageCollection, returns ImageCollection
var evi_images = images.map(addEVI); //apply EVI
var ndwi_images = images.map(addNDWI); //apply NDWI

//apply the bands to images
var images_3bands_extra = images.map(addNDVI).map(addEVI).map(addNDWI);
print('Images with 3 bands added:', images_3bands_extra);

// Apply an extra band to image with minimum cloud cover
// var ndwi_image = addNDWI(image);
// print(ndwi_image);
// print(ndwi_image.date().getRelative('day', 'year'));

// Step 2.6
var mosaic_image = images_3bands_extra.qualityMosaic('NDVI');
//print('mosaic debugging');
print('Mosaic image composed by the largest NDVI pixel values across the ImageCollection:', mosaic_image);

// RYG, red yellow green pallette with red = least vegetation, green = max vegetation
var ndvispectrum = {'min': 0, 'max': 0.5, 'palette': ['orangered','yellow','green'], bands: 'NDVI'};
Map.addLayer(mosaic_image, ndvispectrum, 'Mosaic image with max NDVI pixel values - NDVI band');
Map.addLayer(mosaic_image.clip(rect), ndvispectrum, 'Mosaic image with max NDVI pixel values - NDVI band - rect');

var addDoy = function(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('DOY');
  doyBand = doyBand.updateMask(image.select('NDVI').mask());
  return image.addBands(doyBand);
};

// add DOY (day of year) as a band in the images of the image collection
var images_4bands_extra = images_3bands_extra.map(addDoy); // add the extra band DOY
print('ImageCollection with NDVI, EVI, NDWI and DOY bands:', images_4bands_extra);

// get the image with max NDVI again, but now it will include also the DOY band
var mosaic_image_with_doy = images_4bands_extra.qualityMosaic('NDVI');
print('Mosaic image composed by the largest NDVI pixel values across the ImageCollection and with DOY band:', 
      mosaic_image_with_doy);
var max_ndvi_doy_spectrum = {'min': 1, 'max': 366, 
  'palette': ['yellow', 'lightgreen', 'green', 'red'], bands: 'DOY'};

Map.addLayer(mosaic_image_with_doy, max_ndvi_doy_spectrum, 'DOY (day out of 365) with the max NDVI per pixel');
Map.addLayer(mosaic_image_with_doy.clip(rect), max_ndvi_doy_spectrum, 'DOY (day out of 365) with the max NDVI per pixel - rect');

// Step 2.7
// get all images until today
var images_all = l8.filterBounds(polygon).filter(ee.Filter.lt('CLOUD_COVER', 20)); //.sort("CLOUD_COVER");
print('There are a total of:', images_all.size(), 'images with cloud_cover < 20% until today for the selected area');
var ndvi_images_all = images_all.map(addNDVI); //add NDVI band

// Plot a time series of NDVI at a single location (ROI - region of interest)
var l8Chart = ui.Chart.image.series(ndvi_images_all.select('NDVI'), polygon)
    .setChartType('ScatterChart')
    .setOptions({
      title: 'Landsat 8 NDVI time series at ROI (selected polygon)',
      lineWidth: 1,
      pointSize: 3,
    });
print(l8Chart); // show the graph in the GEE Console


// LINEAR REGRESSION - Plot the same time series along with the fitted Linear Regression line
var l8Chart_Linear_Regression = ui.Chart.image.series(ndvi_images_all.select('NDVI'), polygon)
    .setChartType('ScatterChart')
    .setOptions({
      title: 'Linear Regression model: original and fitted (red line) values',
      trendlines: {0: {
        color: 'CC0000'
      }},
      lineWidth: 1,
      pointSize: 3,
    });
print(l8Chart_Linear_Regression); // show the graph in the GEE Console
// from the above we can see the fit is not appropriate -- the linear model
// cannot explain properly the data


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
      title: 'Harmonic model: original and fitted values',
      lineWidth: 1,
      pointSize: 3,
}));


// Step 2.8: Classification with CART
var images_all_big_polygon = l8.filterBounds(polygon2).filter(ee.Filter.lt('CLOUD_COVER', 20)); //.sort("CLOUD_COVER");
print('There are a total of:', images_all_big_polygon.size(), 
  'images with cloud_cover < 20% until today for the selected area of the big polygon (polygon2)');
var image_big_polygon_with_min_cc = images_all_big_polygon.sort("CLOUD_COVER").first();
print('Here is the image with the lowest cloud cover that contains the polygon 2:', image_big_polygon_with_min_cc)

Map.addLayer(image_big_polygon_with_min_cc, RGB432, "RGB432 for CART - True Color"); // add true RGB color layer to map

//Training data creation
var label = 'Class';
var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
              'B7', 'B8', 'B9', 'B10', 'B11', ];
var input = image_big_polygon_with_min_cc.select(bands); //select bands

var training = Urban.merge(Streets).merge(Water).merge(Vegetation);
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

//cart classification model
var classifier = ee.Classifier.smileCart().train(trainSet, label, bands);

//classify the image
var classified = input.classify(classifier);
print('Classified image:', classified.getInfo());

var landcoverPalette = [
  '#0c2c84', //water (class 0)
  '#e31a1c', //urban (class 1)
  '#005a32', //vegetation (class 2)
  '#808080', //streets (class 3)
  ];
  
Map.addLayer(classified, {palette: landcoverPalette, min:0, max:3}, 'classification');
Map.addLayer(classified.clip(rect), {palette: landcoverPalette, min:0, max:3}, 'classification - rect');

//accuracy assessment
var confusionMatrix = ee.ConfusionMatrix(testSet.classify(classifier)
  .errorMatrix({
    actual: 'Class',
    predicted: 'classification'
  }));

  
print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());


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

// map with max ndvi values and map with doy values are already implemented
// Repeat below process with 2018 data and compare --- what differences can be spotted?







// --------------------------------------------------
// QUESTION 4
// Για 4 διαφορετικά πολύγωνα στην περιοχή μελέτης, για όλα τα διαθέσιμα 
// δεδομένα L8 και τις χρονοσειρές NDVI (ή άλλων δεικτών για τον αστικό χώρο,
// τα ύδατα, το έδαφος), περιγράψτε και παρουσιάστε τις βέλτιστες αρμονικές
// καμπύλες που προσαρμόσατε δημιουργώντας συνθετικές χρονοσειρές.
// --------------------------------------------------




// --------------------------------------------------
// QUESTION 5
// Για την ίδια ή διαφορετική περιοχή μελέτης, ψηφιοποιήστε διάφορες χρήσεις γης
// και πραγματοποιείστε ταξινόμηση στην διαθέσιμη χρονοσειρά. Χρησιμοποιείστε δυο
// διαφορετικούς ταξινομητές. Περιγράψτε τυχόν διαφορές που προκύπτουν στα 
// αποτελέσματα της ταξινόμησης σε σχέση με τις διαθέσιμες ψηφιοποιημένες 
// περιοχές ή τους διαφορετικούς ταξινομητές.
// --------------------------------------------------