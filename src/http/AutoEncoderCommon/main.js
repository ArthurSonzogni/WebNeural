Module = {};

function CreateImageArray(size) {
  const offset = Module._malloc(size);
  Module.HEAPU8.fill(0, offset, offset + size);
  return {
    get data() {
      return Module.HEAPU8.subarray(offset, offset + size);
    },
    offset: offset
  };
}

function CreateInputArray(size) {
  const offset = Module._malloc(size * 8);
  Module.HEAPF64.fill(0, offset / 8, offset / 8 + size);
  return {
    get data() {
      return Module.HEAPF64.subarray(offset / 8, offset / 8 + size);
    },
    offset: offset
  };
}

function DrawImage(canvas, array) {
  const ctx = canvas.getContext('2d');
  if (canvas.width !== 27 || canvas.height !== 27) {
    canvas.width = 27;
    canvas.height = 27;
  }
  const idata = ctx.createImageData(27, 27);
  idata.data.set(array);
  ctx.putImageData(idata, 0, 0);
}

function rand01() {
  return Math.round(Math.random() * 10) / 10;
}

function Slider(container, onInput) {
  const input = document.createElement("input");
  input.type = "range";
  input.min = 0;
  input.max = 1.0;
  input.step = 0.05;
  input.value = 0.5;
  input.addEventListener("input", onInput);
  input.classList.add("v-slider");

  const wrap = document.createElement("div");
  wrap.classList.add("slider_container");
  wrap.appendChild(input);
  container.appendChild(wrap);
  return input;
}

function SliderGroup(container) {
  const object = {
    sliders: [],
    values: function() {
      return this.sliders.map(it => parseFloat(it.value));
    },
    setValues: function(array) {
      this.sliders.forEach(function(slider, i) {
        slider.value = array[i];
      });
    },
    randomize: function() {
      this.setValues(Array.from({ length: 10 }, () => rand01()));
    }
  };
  for (let i = 0; i < 10; ++i) {
    object.sliders.push(Slider(container, UpdateImages));
  }
  return object;
}

let can_train = false;
let train_steps = 0;
let is_animating = true;
let anim_phase = 0;

function SetTrainStatus(text) {
  const el = document.querySelector("#train_status");
  if (el) el.textContent = text;
}

function Next() {
  TrainJS();
}

function ToogleTraining() {
  can_train = !can_train;
  const btn = document.querySelector("#toogle_training");
  if (can_train) {
    btn.innerText = "Stop Training";
    SetTrainStatus("Training in browser... (" + train_steps + " steps)");
  } else {
    btn.innerText = "Start Training";
    SetTrainStatus("Paused (" + train_steps + " steps)");
  }
}

function ToggleAnimate() {
  is_animating = !is_animating;
  document.querySelector("#toggle_animate").innerText =
      is_animating ? "Pause Morph" : "Play Morph";
}

const container_a = document.querySelector("#container_a");
const container_b = document.querySelector("#container_b");
const slider_lambda = document.querySelector("#slider_lambda");
slider_lambda.addEventListener("input", UpdateImages);

const sliders_a = SliderGroup(container_a);
const sliders_b = SliderGroup(container_b);

sliders_a.setValues([0.6, 0.1, 0.3, 0.1, 0.7, 0.5, 0.3, 0.4, 0.2, 0.4]);
sliders_b.setValues([0.8, 0.0, 0.6, 0.8, 0.6, 0.7, 0.2, 0.6, 0.2, 0.4]);

// 9-step interpolation filmstrip
const strip_canvases = [];
const strip_el = document.querySelector("#strip");
for (let s = 0; s < 9; ++s) {
  const t = s / 8;
  const wrap = document.createElement("div");
  wrap.className = "strip-item";
  const c = document.createElement("canvas");
  c.title = "Set interpolation λ = " + t.toFixed(2);
  c.addEventListener("click", () => {
    slider_lambda.value = t;
    UpdateImages();
  });
  const lbl = document.createElement("span");
  lbl.textContent = s === 0 ? "A" : (s === 8 ? "B" : Math.round(t * 100) + "%");
  wrap.appendChild(c);
  wrap.appendChild(lbl);
  strip_el.appendChild(wrap);
  strip_canvases.push({ canvas: c, t: t });
}

// 16-item random gallery
const gallery_items = [];
const gallery_el = document.querySelector("#gallery");
for (let g = 0; g < 16; ++g) {
  const c = document.createElement("canvas");
  const item = { canvas: c, latent: new Array(10).fill(0.5) };
  c.addEventListener("click", (e) => {
    if (e.shiftKey) {
      sliders_b.setValues(item.latent);
    } else {
      sliders_a.setValues(item.latent);
    }
    UpdateImages();
  });
  c.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    sliders_b.setValues(item.latent);
    UpdateImages();
  });
  gallery_el.appendChild(c);
  gallery_items.push(item);
}

function UpdateImages() {
  if (!Predict) return;
  const values_a = sliders_a.values();
  const values_b = sliders_b.values();
  const lambda = parseFloat(slider_lambda.value);

  for (let i = 0; i < 10; ++i) latent.data[i] = values_a[i];
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_a"), output.data);

  for (let i = 0; i < 10; ++i) latent.data[i] = values_b[i];
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_b"), output.data);

  for (let i = 0; i < 10; ++i) {
    latent.data[i] = values_a[i] * (1.0 - lambda) + values_b[i] * lambda;
  }
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_c"), output.data);

  for (const step of strip_canvases) {
    for (let i = 0; i < 10; ++i) {
      latent.data[i] = values_a[i] * (1.0 - step.t) + values_b[i] * step.t;
    }
    Predict(latent.offset, output.offset);
    DrawImage(step.canvas, output.data);
  }
}

function SampleGallery() {
  if (!Predict) return;
  for (const item of gallery_items) {
    for (let i = 0; i < 10; ++i) {
      const z = rand01();
      item.latent[i] = z;
      latent.data[i] = z;
    }
    Predict(latent.offset, output.offset);
    DrawImage(item.canvas, output.data);
  }
}

function RandomizeA() {
  sliders_a.randomize();
  UpdateImages();
}

function RandomizeB() {
  sliders_b.randomize();
  UpdateImages();
}

function ResetModel() {
  ResetModelWeight();
  train_steps = 0;
  SetTrainStatus("Weights reset (untrained)");
  TrainJS();
  SampleGallery();
}

function LoadOptimizedParams() {
  LoadPretrainedModel();
  train_steps = 0;
  SetTrainStatus("Pretrained model loaded");
  TrainJS();
  SampleGallery();
}

let latent, input, output;
let Train, Predict, LastInput, LastOutput, LoadPretrainedModel, ResetModelWeight;

Module.onRuntimeInitialized = function() {
  Predict = Module.cwrap('Predict', 'void', ['number', 'number']);
  Train = Module.cwrap('Train', 'void', []);
  LastInput = Module.cwrap('LastInput', 'void', ['number']);
  LastOutput = Module.cwrap('LastOutput', 'void', ['number']);
  LoadPretrainedModel = Module.cwrap('LoadPretrainedModel', 'void', []);
  ResetModelWeight = Module.cwrap('ResetModelWeight', 'void', []);

  input = CreateImageArray(27 * 27 * 4);
  output = CreateImageArray(27 * 27 * 4);
  latent = CreateInputArray(10);

  document.querySelector("#loading").className = "disabled";
  LoadOptimizedParams();
};

function TrainJS() {
  Train();
  train_steps += 1;
  if (can_train) {
    SetTrainStatus("Training in browser... (" + train_steps + " steps)");
  }
  LastInput(input.offset);
  LastOutput(output.offset);
  DrawImage(document.querySelector("#input"), input.data);
  DrawImage(document.querySelector("#output"), output.data);
  UpdateImages();
}

function UpdateMorph() {
  if (!Predict) return;
  const values_a = sliders_a.values();
  const values_b = sliders_b.values();
  const lambda = parseFloat(slider_lambda.value);
  for (let i = 0; i < 10; ++i) {
    latent.data[i] = values_a[i] * (1.0 - lambda) + values_b[i] * lambda;
  }
  Predict(latent.offset, output.offset);
  DrawImage(document.querySelector("#image_c"), output.data);
}

setInterval(function() {
  if (can_train) {
    TrainJS();
  }
  if (is_animating) {
    anim_phase += 0.04;
    slider_lambda.value = (0.5 - 0.5 * Math.cos(anim_phase)).toFixed(3);
    UpdateMorph();
  }
}, 1000.0 / 30.0);
