Module = {};

function CreateImageArray(size) {
  var offset = Module._malloc(size);
  Module.HEAPU8.set(new Uint8ClampedArray(size), offset);
  return {
    "data": Module.HEAPU8.subarray(offset, offset + size),
    "offset": offset
  }
}

function CreateInputArray(size) {
  var offset = Module._malloc(size * 8);
  Module.HEAPF64.set(new Float64Array(size), offset / 8);
  return {
    "data": Module.HEAPF64.subarray(offset / 8, offset / 8 + size),
    "offset": offset
  }
}

function DrawImage(canvas, array) {
  // create off-screen canvas element
  let ctx = canvas.getContext('2d');

  canvas.width = 27;
  canvas.height = 27;
  canvas.style.width = "120px";
  canvas.style.height = "120px";

  // create imageData object
  let idata = ctx.createImageData(27, 27);

  // set our buffer as source
  idata.data.set(array);

  // update canvas with new data
  ctx.putImageData(idata, 0, 0);
}

let random_seed = 132222;
function random() {
  for(var i = 0; i<2; ++i) {
    random_seed = (random_seed * 121 + 97) % 562;
  }
  return random_seed;
}
function Slider(container, onInput) {
  let input = document.createElement("input");
  input.type = "range";
  input.min = 0;
  input.max = 1.0;
  input.step = 0.1;
  input.value = (random() % 9) * 0.1 + 0.1;
  input.addEventListener("input", onInput);
  input.classList.add("slider");
  container.appendChild(input);
  return input;
}

function SliderGroup(container) {
  let object = {
    sliders: [],
    values: function() {
      return this.sliders.map(it => it.value);
    },
    onUpdated: function(){},
  }
  for(let i = 0; i<10; ++i){
    object.sliders.push(Slider(container, UpdateImages));
  }
  return object;
}

function Next() {
  TrainJS();
}

let can_train = false;
let StartTraining = function() { can_train = true; }
let StopTraining = function() { can_train = false; }
function ToogleTraining() {
  can_train = !can_train;
  let toogle_training = document.querySelector("#toogle_training");
  if (can_train) {
    toogle_training.innerText = "Stop training";
  } else {
    toogle_training.innerText = "Start training";
  }
}

let lambda = 0.0;

let container_a = document.querySelector("#container_a");
let container_b = document.querySelector("#container_b");
let container_lambda = document.querySelector("#container_lambda");

let sliders_a = SliderGroup(container_a);
let sliders_b = SliderGroup(container_b);
let slider_lambda = Slider(container_c, UpdateImages);

function UpdateImages() {
  let values_a = sliders_a.values();
  let values_b = sliders_b.values();
  let lambda = slider_lambda.value;

  // Update A.
  for(let i = 0; i<10; ++i) {
    latent.data[i] = values_a[i];
  }
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_a"), output.data);

  // Update B.
  for(let i = 0; i<10; ++i) {
    latent.data[i] = values_b[i];
  }
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_b"), output.data);

  // Update C
  for(let i = 0; i<10; ++i) {
  latent.data[i] = values_a[i] * (1.0 - lambda) + values_b[i] * lambda;
  }
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_c"), output.data);
}

function LoadOptimizedParams() {
  LoadPretrainedModel();
  TrainJS();
}

var latent;
var input;
var output;
var image_a;
var image_b;

var Train;
var Predict
var LastInput;
var LastOutput;
var LoadPretrainedModel;

Module.onRuntimeInitialized = function() {
  Predict = Module.cwrap('Predict','number',['number', 'number'])
  Train = Module.cwrap('Train','void',[]);
  LastInput = Module.cwrap('LastInput','void',['number']);
  LastOutput = Module.cwrap('LastOutput','void',['number']);
  LoadPretrainedModel = Module.cwrap('LoadPretrainedModel','void',['number']);

  input = CreateImageArray(27*27*4);
  output = CreateImageArray(27*27*4);
  image_a = CreateImageArray(27*27*4);
  image_b = CreateImageArray(27*27*4);
  latent = CreateInputArray(10);

  TrainJS();
}

function TrainJS() {
  Train();
  LastInput(input.offset);
  LastOutput(output.offset);
  DrawImage(document.querySelector("#input"), input.data);
  DrawImage(document.querySelector("#output"), output.data);
  UpdateImages();
}

setInterval(function() {
  if (can_train)
    TrainJS();
}, 1.0/30.0);
