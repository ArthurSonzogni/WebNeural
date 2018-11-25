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

  canvas.width = 29;
  canvas.height = 29;
  canvas.style.width = "120px";
  canvas.style.height = "120px";

  // create imageData object
  let idata = ctx.createImageData(29, 29);

  // set our buffer as source
  idata.data.set(array);

  // update canvas with new data
  ctx.putImageData(idata, 0, 0);
}

let random_seed = 5643;
function random() {
  for(var i = 0; i<10; ++i) {
    random_seed = (random_seed * 121 + 97) % 9999;
  }
  return random_seed;
}

function Slider(container, onInput) {
  let input = document.createElement("input");
  input.orient = "vertical"; // For firefox.
  input.type = "range";
  input.min = -2.0;
  input.max = 2.0;
  input.step = 0.1;
  input.value = (random() % 9) * 0.2 + 0.2 - 1.0;
  input.addEventListener("input", onInput);
  input.classList.add("slider");

  let input_container = document.createElement("div");
  input_container.classList.add("slider_container");
  input_container.appendChild(input);

  container.appendChild(input_container);
  return input;
}

function SliderGroup(container) {
  let object = {
    sliders: [],
    values: function() {
      return this.sliders.map(it => it.value);
    },
    setValues: function(array) {
      let i = 0; 
      this.sliders.forEach(function(slider) {
        slider.value = (array[i] - 0.5) * 2.0;
        i = i+1;
      });
    },
    onUpdated: function(){},
  }
  for(let i = 0; i<10; ++i){
    object.sliders.push(Slider(container, UpdateImages));
  }
  return object;
}

let can_train = false;
let StartTraining = function() { can_train = true; }
let StopTraining = function() { can_train = false; }
function ToogleTraining() {
  console.log("ToogleTraining");
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

sliders_a.setValues([0.7, 0.1, 0.9, 0.6, 0.2, 0.6, 0, 0.6, 1, 0.4]);
sliders_b.setValues([0.5, 0.1, 0.4, 0.5, 0.2, 0.8, 0, 0.4, 1, 0.1]);

function UpdateImages() {
  let values_a = sliders_a.values();
  let values_b = sliders_b.values();
  let lambda = slider_lambda.value * 0.25 + 0.5;

  // Update A.
  let A_XX = 0.0;
  for(let i = 0; i<10; ++i) {
    input.data[i] = values_a[i];
    A_XX += values_a[i] * values_a[i];
  }
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_a"), output.data);

  // Update B.
  let B_XX = 0.0;
  for(let i = 0; i<10; ++i) {
    input.data[i] = values_b[i];
    B_XX += values_b[i] * values_b[i];
  }
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_b"), output.data);

  // Update C
  let C_XX = 0.0;
  for(let i = 0; i<10; ++i) {
    let interpolation = values_a[i] * (1.0 - lambda) + values_b[i] * lambda;
    input.data[i] = interpolation;
    C_XX += interpolation * interpolation;
  }
  let ratio = (Math.sqrt(A_XX) * (1.0 - lambda) + Math.sqrt(B_XX) * lambda) / Math.sqrt(C_XX);
  for(let i = 0; i<10; ++i) {
    input.data[i] *= ratio;
  }
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_c"), output.data);
}

// Data Predict(input -> output);
let Predict
let input, output;

// Improve Predict.
let Train

// Reset/Load model
let LoadPretrainedModel;
let ResetModelWeight;

function ResetModel() {
  ResetModelWeight();
  UpdateImages();
}

function LoadOptimizedParams() {
  LoadPretrainedModel();
  UpdateImages();
}

Module.onRuntimeInitialized = function() {
  Predict = Module.cwrap('Predict','number',['number', 'number'])
  LoadPretrainedModel = Module.cwrap('LoadPretrainedModel','number',['number', 'number'])
  Train = Module.cwrap('Train','void',[]);
  ResetModelWeight = Module.cwrap('ResetModelWeight','void',[]);

  input = CreateInputArray(10);
  output = CreateImageArray(29*29*4);

  document.querySelector("#loading").className = "disabled";
  LoadOptimizedParams();
}

function TrainJS() {
	console.log("TrainJS");
  Train();
  UpdateImages();
}

setInterval(function() {
  if (can_train)
    TrainJS();
}, 1.0/30.0);
