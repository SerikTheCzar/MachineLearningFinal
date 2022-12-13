const numeric = require('numeric');
const math = require('mathjs');
const { print } = require('mathjs');
function ML(generations){
function sigmoid(z) {
 //  console.log("this is z: " + z);
  //  console.log("This is the math function:" + (1/(1+Math.pow(Math.E, -z))))
    //console.log(Math.pow(Math.E, -z));
    return 1/(1+Math.pow(Math.E, -z));
   
}
//console.log("SIGMOID TEST ON NEGATIVE : " + sigmoid(-1.8522557025943267));
function tanh(z) {
    let numerator = Math.pow(Math.E, 2*z) - 1;
    let denominator = Math.pow(Math.E, 2*z) + 1;
    return numerator/denominator;
}
function ReLu(z) {
    return Math.max(0,z);
}

function LeReLu(z) {
    return Math.max(0.1*z, z);
}

function binStep(z) {
    
    if(x<0) {
        return 0
    } else{
        return 1
    }
}

function initParam(inputFeatures, numHiddenLayers, outputFeatures) {
    let W1 = Array.from(new Array(numHiddenLayers), _ => Array(inputFeatures).fill(Math.random()));//Math.random() * (numHiddenLayers - inputFeatures) + numHiddenLayers;
    let W2 = Array.from(new Array(outputFeatures), _ => Array(numHiddenLayers).fill(Math.random()));//Math.random() * (outputFeatures - numHiddenLayers) + outputFeatures;
    let b1 = Array(2).fill(0).map(x => [x]);

 //   console.log("this is w1: " + W1);
    //console.log("b1 is: " + b1);
    let b2 = Array(1).fill(0).map(x => [x]);


    //console.log("b2 is: " + b2);
    let parameters = {"W1" : W1, "b1": b1, "W2" : W2, "b2": b2};
  //  console.log(parameters);
    return parameters;
    
} //this works
//initParam(2, 2, 1);
function Activation(Z1) {
  ReLu(Z1);
    //solve this later
}
function crossDotButCooler(X, Y) {
 //   console.log("this is x sub 1: " + X[0]);
  //  console.log("this is y: " + Y[0][0]); 
    let matrix = Array.from({length: 1}, () => Array.from({length: 4}, () => 0));
  //  console.log("this is matrix: " + matrix);
    matrix[0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0];
   // console.log("this is matrix[0]: " + matrix[0]);
    matrix[1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1];
    matrix[2] = X[0][0]*Y[0][2] + X[0][1]*Y[1][2];
    matrix[3] = X[0][0]*Y[0][3] + X[0][1]*Y[1][3];

    return matrix;
}
function crossDot(X, Y) {
    const matrix = Array.from({length: 2}, () => Array.from({length: 4}, () => 0));
    matrix[0][0] = X[0][0] * Y[0][0] + X[0][1] * Y[1][0]; 
    matrix[0][1] = X[0][0] * Y[0][1] + X[0][1] * Y[1][1];
    matrix[0][2] = X[0][0] * Y[0][2] + X[0][1] * Y[1][2];
    matrix[0][3] = X[0][0] * Y[0][3] + X[0][1] * Y[1][3];
    matrix[1][0] = X[1][0] * Y[0][0] + X[1][1] * Y[1][0];
    matrix[1][1] = X[1][0] * Y[0][1] + X[1][1] * Y[1][1];
    matrix[1][2] = X[1][0] * Y[0][2] + X[1][1] * Y[1][2];
    matrix[1][3] = X[1][0] * Y[0][3] + X[1][1] * Y[1][3];

//   console.log("this is matrix: " + matrix);
    return matrix;
    
//2x4 matrix in javascript
}
function forwardProps(X,Y,parameters, funType) {
    let m = X[1].length;
   
    let W1 = parameters["W1"];
    let W2 = parameters["W2"];
    let b1 = parameters["b1"];
    let b2 = parameters["b2"];
   // console.log("this is x: " + X);
    /*
    console.log("this is W1: " + W1);    
    console.log("this is W2: " + W2);
  * console.log("this is b1: " + b1);  
    console.log("this is b2: " + b2);    
*/ //these all match
    let zl =  Array.from({length: 2}, () => Array.from({length: 2}, () => 0));
    zl[0][0] = -0.98014093;
    zl[0][1] = 1.34730991;
    zl[1][0] = 0.23596399;
    zl[1][1] = -1.49260476;

  //  console.log("this is X length: " + X);
   // console.log("this is w length: " + W1);
    //console.log("helping figure out crossdot: this is W1: " + W1);
    let Z1 = crossDot(W1, X);
   // console.log("this is Z1's data type: " + typeof(Z1[0][1]));
    //these match? do 
    let A1=Array.from({length: 2}, () => Array.from({length: 4}, () => 0));
    for(let i = 0; i < Z1.length; i++) {
      for(let j = 0; j < Z1[0].length; j++) {
      //  console.log("dimmensions of Z1: " + Z1.length + " " + Z1[0].length);
        Z1[i][j] += b1[0][0];
        //console.log("sigmoid on a test value: " + sigmoid(-0.01165403056822295));
        let fv = Z1[i][j];
        A1[i][j] = sigmoid(fv);
      //  console.log("this is the sigmoid on Z1: " + sigmoid(fv));
      }
    }
 //   console.log("This is A1: " + A1);
   // console.log("this is Z1's datatype: " + typeof(Z1[0]));
 //   console.log("line 54");
    //let A1 = [[]];
  //  Z1.forEach(element => A1.push(element.forEach(element => A1.push(sigmoid(element)))));
   
    // console.log("this is A1: " + A1 + "]");
  //  console.log("This is A1: " + A1);
    //console.log("this is A1: " + A1);
    //console.log("this is W2: " + W2);
   // console.log("this is the W2 input: " + W2);
    //console.log("this is the X input: " + X);
   // console.log("this is the W2 input for crossDotBC: " + W2.length); 
   // console.log("this is the A1 input for crossDotBC: " + X.length);
    let Z2 = crossDotButCooler(W2, A1);// + b2;
   // console.log("proper Z2: " + Z2);
   // console.log("Z2 before the surgery: " + Z2);
   // console.log(b2[0][0]);
   let A2 = [];
    for(let i = 0; i < Z2.length; i++) {
     // console.log("this is b2's data type: " + typeof(b2[0][0]))
      Z2[i] += b2[0][0];
      let fr = Z2[i];
      A2[i] = sigmoid(fr);
    }

   // console.log("this is Z2 length now: " + Z2.length);
    //   console.log("line 58");
  
    
   // Z2.forEach(element => A2.push(sigmoid(element)));

    let cache = [];//(Z1, A1, W1, b1, Z2, A2, W2, b2);
    cache.push(Z1);
    cache.push(A1);
    cache.push(W1);
    cache.push(b1);
    cache.push(Z2);
    cache.push(A2);
    cache.push(W2);
    cache.push(b2);
    /*
    console.log("This is Z1: " + Z1); //Z1 matches
    console.log("This is b1: " + b1); //b1 matches
    console.log("This is Z2: " + Z2); //Z2 match
    console.log("This is b2: " + b2); //b2 matches
    console.log("This is A1: " + A1); //A1 matches
    console.log("This is A2: " + A2); //A2 matches
  */
    //  console.log("this is cache: " + cache);
    //let log = numeric.add(numeric.mul(numeric.log(A2), Y), numeric.mul(numeric.log(numeric.sub(1, A2)), numeric.sub(1, Y)));
   // console.log("here lies the problem");
  //  console.log("this is A2: " + A2);
    let mLog = Math.log(A2);
   // console.log("this is mLog: " + mLog);
    var log = numeric.add(numeric.mul(numeric.log(A2), Y), numeric.mul(numeric.log(numeric.sub(1, A2)), numeric.sub(1, Y)));


 //  console.log("line 62");
    let precost = 0;
    for(let b=0;b<log.length;b++) {
        precost+=log[b];
    }
   
  //  console.log("flog: " + precost);
    let cost = precost/m;
    let pack=[];
    pack.push(cost);
    pack.push(cache);
    pack.push(A2);
    return pack;
    
    
} //not tested whatsoever
function sub(x, y) {
  let resT =[];
  for(let i=0;i<x.length;i++) {
    resT[i] = x[i]-y[0][i];
  }
 // console.log("Rest value: " + resT);
return resT;
}
function multiply(a, b) {
  var aNumRows = a.length, aNumCols = a[0].length,
      bNumRows = b.length, bNumCols = b[0].length,
      m = new Array(aNumRows);  // initialize array of rows
  for (var r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols); // initialize the current row
    for (var c = 0; c < bNumCols; ++c) {
      m[r][c] = 0;             // initialize the current cell
      for (var i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
}
function sumDif(X){
  let ret =  0;
  //console.log("this is X: " + X[1]);
  for(let i=0;i<X.length;i++) {
    ret+= X[i];
  }
 // console.log(ret);
  let rt2 = [[]];
  rt2[0][0] = ret;
 // console.log("this is rt2: " + rt2);
  return rt2;

}

function multiplyMatrices(X, Y) {

}
function backwardPropagation(X, Y, cache) {
    var m = X[1].length;
   // console.log("this is cache: " + cache);
    var Z1 = cache[0];
    var A1 = cache[1];
    var W1 = cache[2];
    var b1 = cache[3];
    var Z2 = cache[4];
    var A2 = cache[5];
    var W2 = cache[6];
    var b2 = cache[7];
 //   console.log("Cache values: " + Z1 + " A1 val" + A1+ " W1 Val" + W1+ " B1 valu" + b1);
  //console.log("this is y dimensions: " + Y.length); 
  //console.log("this is A2 dimensions: " + A2.length); 
  let ffe = sub(A2, Y);
  //perserve dimensions:

  let frs = [[]];
  for(let b = 0;b<ffe.length;b++) {
    frs[0][b] = ffe[b];
  }
  //console.log("perseving dimensions: " + frs);
  var dZ2 = 3;//sub(A2, Y); <----
  //console.log("this is dZ2 dimensions: " + ffe[0].length);
   // console.log("DZ2:::" + Y);
   // console.log("now this is A1: " + A1);
    var dW2 = multiply(frs, numeric.transpose(A1));// / m;
    
    for(let i=0;i<dW2[0].length;i++) {
      dW2[0][i] = dW2[0][i]/m;
    }

    //use frs instead of dZ2
   // console.log("this is dW2: " + dW2);
    //console.log("ffe dimensions: " + ffe.length + " " + ffe[0].length);
   // console.log("A1 transpose dimensions: " + numeric.transpose(A1).length + " " + numeric.transpose(A1)[0].length);
 //   console.log("this is djalen: " + dW2);
    //var dW2 = numeric.div(numeric.dot(dZ2, numeric.transpose(A1)), m);
   // console.log("it's 99");
    var db2 = sumDif(ffe);// numeric.div(numeric.sum(dZ2, 1), m);
    /*
    console.log("this is dZ2: " + ffe);
    console.log("this is dW2: " + dW2);
    console.log("this is db2: " + db2); 
    */
  //  console.log("it's 99");
   // console.log("this is W2 dimensions after transpose: " + numeric.transpose(W2).length + " " + numeric.transpose(W2)[0].length);
    //console.log("this is dZ2: " + dZ2);
    var dA1 = multiply(numeric.transpose(W2), frs);
    
    //  console.log("it's 121");
    let a1sub = numeric.sub(1, A1);
   // console.log("this is a1sub: " + a1sub);
    let A1square = numeric.mul(A1, a1sub);
 //   console.log("this is A1square: " + A1square);
    var dZ1 = numeric.mul(dA1, A1square);
    //var dZ1 = numeric.mul(dA1, numeric.mul(A1, numeric.sub(1, A1)));
 //   console.log("it's 122");
 //   console.log("dA1 value" + dA1);
    var dW1 = numeric.div(multiply(dZ1, numeric.transpose(X)), m);
 //   console.log("it's 125");
 
    var db1 = numeric.div(numeric.sum(dZ1, 1), m);
  /*
    console.log("this is dA1: " + dA1);
    console.log("this is dZ1: " + dZ1);
    console.log("this is dW1: " + dW1);
    console.log("this is db1: " + db1);  
*/
    var gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2, "dZ1": dZ1, "dW1": dW1, "db1": db1};
    return gradients;
}

function updateParameters(parameters, gradients, learningRate) {
    parameters["W1"] = numeric.sub(parameters["W1"], numeric.mul(learningRate, gradients["dW1"]));
    parameters["W2"] = numeric.sub(parameters["W2"], numeric.mul(learningRate, gradients["dW2"]));
    parameters["b1"] = numeric.sub(parameters["b1"], numeric.mul(learningRate, gradients["db1"]));
    parameters["b2"] = numeric.sub(parameters["b2"], numeric.mul(learningRate, gradients["db2"]));
    return parameters;
}
var X = [[0, 0, 1, 1], [0, 1, 0, 1]]; // XOR input
var Y = [[0, 1, 1, 0]]; // XOR output

// Define model parameters
var neuronsInHiddenLayers = 2; // number of hidden layer neurons (2)
var inputFeatures = X.length;
//console.log(inputFeatures); // number of input features (2)
var outputFeatures = Y.length; // number of output features (1)
var parameters = initParam(inputFeatures, neuronsInHiddenLayers, outputFeatures);
//console.log("after parameters");
var epoch = generations;
var learningRate = 0.01;
var losses = Array(epoch).fill(0);
for (var i = 0; i < epoch; i++) {
  //  console.log("wog");
    var result = forwardProps(X, Y, parameters);
    losses[i][0] = result[0];
   // console.log("losses: " + losses[i][0]);
    var cache = result[1];
   // console.log("cache value: $ " + cache);
    var A2 = result[2];
   // console.log("line cache: " + result);
    var gradients = backwardPropagation(X, Y, cache);
    parameters = updateParameters(parameters, gradients, learningRate);
 //   console.log("thjis is parameters: " + parameters);
}


var X = [[0, 0, 1, 1], [0, 1, 0, 1]]; // XOR input
var result = forwardProps(X, Y, parameters);
var cost = result[0];
var A2 = result[2];
//let prediction = (A2 > 0.5) * 1.0;
var prediction = numeric.mul(numeric.gt(A2, 0.5), 1.0);
//console.log("this is wehre teh error's happening" + A2);
let sysCheckList = [];
sysCheckList.push("A1 is working");
sysCheckList.push("A2 is working");
sysCheckList.push("Z1 is working");
sysCheckList.push("Z2 is working");
sysCheckList.push("B1 is working");
console.log(prediction);
return prediction;
}

const t0 = performance.now();
let accuracy = ML(1000000);
let initAccuracy = 0;
let answerKey = [0, 1, 1, 0];
for(let i=0;i<4; i++) {
  if(accuracy[i] == answerKey[i]) {
    initAccuracy++;
  }
}
const t1 = performance.now();
let percentScore = (initAccuracy/4)*100;
console.log(`The machine learning algorithm took ${t1 - t0} milliseconds with an accuracy of ${percentScore}%.`);