Module = {};

function CreateImageArray(size) {
  const offset = Module._malloc(size);
  Module.HEAPU8.set(new Uint8ClampedArray(size), offset);
  return {
    data: Module.HEAPU8.subarray(offset, offset + size),
    offset: offset
  };
}

function CreateInputArray(size) {
  const offset = Module._malloc(size * 8);
  Module.HEAPF64.set(new Float64Array(size), offset / 8);
  return {
    data: Module.HEAPF64.subarray(offset / 8, offset / 8 + size),
    offset: offset
  };
}

function DrawImage(canvas, array) {
  const ctx = canvas.getContext('2d');
  if (canvas.width !== 29 || canvas.height !== 29) {
    canvas.width = 29;
    canvas.height = 29;
  }
  const idata = ctx.createImageData(29, 29);
  idata.data.set(array);
  ctx.putImageData(idata, 0, 0);
}

// Box-Muller Gaussian N(0, 1), clamped to slider range [-2.5, 2.5].
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return Math.max(-2.5, Math.min(2.5, Math.round(z * 10) / 10));
}

function Slider(container, onInput) {
  const input = document.createElement("input");
  input.type = "range";
  input.min = -2.5;
  input.max = 2.5;
  input.step = 0.1;
  input.value = 0;
  input.addEventListener("input", onInput);
  input.classList.add("v-slider");

  const input_container = document.createElement("div");
  input_container.classList.add("slider_container");
  input_container.appendChild(input);

  container.appendChild(input_container);
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
      const vals = Array.from({ length: 10 }, () => randn());
      this.setValues(vals);
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
  const btn = document.querySelector("#toggle_animate");
  btn.innerText = is_animating ? "Pause Morph" : "Play Morph";
}

const container_a = document.querySelector("#container_a");
const container_b = document.querySelector("#container_b");
const slider_lambda = document.querySelector("#slider_lambda");
slider_lambda.addEventListener("input", UpdateImages);

const sliders_a = SliderGroup(container_a);
const sliders_b = SliderGroup(container_b);

sliders_a.setValues([0.4, -0.8, 0.9, 0.2, -0.6, 0.5, -1.0, 0.3, 1.1, -0.2]);
sliders_b.setValues([-0.6, 0.7, -0.4, 0.8, 0.5, -0.9, 0.4, -0.7, -0.3, 0.9]);

// Build 9-step interpolation filmstrip canvases.
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

// Build 16-item random sample gallery.
const gallery_items = [];
const gallery_el = document.querySelector("#gallery");
for (let g = 0; g < 16; ++g) {
  const c = document.createElement("canvas");
  const item = { canvas: c, latent: new Array(10).fill(0) };
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

function InterpolateSpherical(values_a, values_b, lambda, dest) {
  let A_XX = 0.0, B_XX = 0.0, C_XX = 0.0;
  for (let i = 0; i < 10; ++i) {
    A_XX += values_a[i] * values_a[i];
    B_XX += values_b[i] * values_b[i];
    const interp = values_a[i] * (1.0 - lambda) + values_b[i] * lambda;
    dest[i] = interp;
    C_XX += interp * interp;
  }
  if (C_XX > 1e-8) {
    const ratio = (Math.sqrt(A_XX) * (1.0 - lambda) + Math.sqrt(B_XX) * lambda) / Math.sqrt(C_XX);
    for (let i = 0; i < 10; ++i) {
      dest[i] *= ratio;
    }
  }
}

function UpdateImages() {
  if (!Predict) return;
  const values_a = sliders_a.values();
  const values_b = sliders_b.values();
  const lambda = parseFloat(slider_lambda.value);

  // Update A
  for (let i = 0; i < 10; ++i) input.data[i] = values_a[i];
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_a"), output.data);

  // Update B
  for (let i = 0; i < 10; ++i) input.data[i] = values_b[i];
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_b"), output.data);

  // Update C
  InterpolateSpherical(values_a, values_b, lambda, input.data);
  Predict(input.offset, output.offset);
  DrawImage(document.querySelector("#image_c"), output.data);

  // Update 9-step strip
  for (const step of strip_canvases) {
    InterpolateSpherical(values_a, values_b, step.t, input.data);
    Predict(input.offset, output.offset);
    DrawImage(step.canvas, output.data);
  }
}

function SampleGallery() {
  if (!Predict) return;
  for (const item of gallery_items) {
    for (let i = 0; i < 10; ++i) {
      const z = randn();
      item.latent[i] = z;
      input.data[i] = z;
    }
    Predict(input.offset, output.offset);
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

let Predict;
let input, output;
let Train;
let LoadPretrainedModel;
let ResetModelWeight;

function ResetModel() {
  ResetModelWeight();
  train_steps = 0;
  SetTrainStatus("Model weights reset");
  UpdateImages();
  SampleGallery();
}

function LoadOptimizedParams() {
  LoadPretrainedModel();
  train_steps = 0;
  SetTrainStatus("Pretrained model loaded");
  UpdateImages();
  SampleGallery();
}

Module.onRuntimeInitialized = function() {
  Predict = Module.cwrap('Predict', 'void', ['number', 'number']);
  LoadPretrainedModel = Module.cwrap('LoadPretrainedModel', 'void', []);
  Train = Module.cwrap('Train', 'void', []);
  ResetModelWeight = Module.cwrap('ResetModelWeight', 'void', []);

  input = CreateInputArray(10);
  output = CreateImageArray(29 * 29 * 4);

  document.querySelector("#loading").className = "disabled";
  LoadOptimizedParams();
};

function TrainJS() {
  Train();
  train_steps += 1;
  SetTrainStatus("Training in browser... (" + train_steps + " steps)");
  UpdateImages();
}

function UpdateMorph() {
  if (!Predict) return;
  const values_a = sliders_a.values();
  const values_b = sliders_b.values();
  const lambda = parseFloat(slider_lambda.value);
  InterpolateSpherical(values_a, values_b, lambda, input.data);
  Predict(input.offset, output.offset);
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
