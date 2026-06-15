// BaselineSNNPresentation.pde
// Animated project visualization for SNN-on-GPU baseline comparison
// Processing Java Mode

ArrayList<FlowParticle> particles = new ArrayList<FlowParticle>();
ArrayList<Spike> spikes = new ArrayList<Spike>();

float time = 0;
float membrane = 0;
float threshold = 0.75;
float resetValue = 0.18;

PFont fontMain;

void setup() {
  size(1500, 850);
  smooth(8);
  frameRate(60);

  fontMain = createFont("Arial", 18);
  textFont(fontMain);

  for (int i = 0; i < 90; i++) {
    particles.add(new FlowParticle());
  }

  for (int i = 0; i < 45; i++) {
    spikes.add(new Spike());
  }
}

void draw() {
  background(248, 250, 252);
  time += 0.015;

  drawTitle();
  drawMainPipeline();
  drawBaselineBranches();
  drawSNNMembranePlot();
  drawMetricsDashboard();
  drawFooterMessage();

  for (FlowParticle p : particles) {
    p.update();
    p.display();
  }

  for (Spike s : spikes) {
    s.update();
    s.display();
  }
}

// ------------------------------------------------------------
// TITLE
// ------------------------------------------------------------

void drawTitle() {
  fill(15, 23, 42);
  textAlign(CENTER);
  textSize(34);
  text("Baseline Comparison for GPU-Based SNN Simulation", width / 2, 48);

  textSize(17);
  fill(71, 85, 105);
  text("MLP, CNN, RNN/LSTM, and SNN evaluated under the same event-data workflow, hardware, and metrics", width / 2, 78);
}

// ------------------------------------------------------------
// MAIN PIPELINE
// ------------------------------------------------------------

void drawMainPipeline() {
  float y = 155;
  float w = 190;
  float h = 78;
  float gap = 35;

  String[] labels = {
    "Event Dataset",
    "Event Workflow",
    "Baseline Models",
    "GPU Profiling",
    "Visualization",
    "Conclusion"
  };

  String[] sub = {
    "N-MNIST / DVS",
    "cache + slicing",
    "MLP CNN RNN SNN",
    "latency + memory",
    "plots + dashboard",
    "deployment viability"
  };

  for (int i = 0; i < labels.length; i++) {
    float x = 60 + i * (w + gap);
    drawBlock(x, y, w, h, labels[i], sub[i], color(255), color(14, 165, 233));

    if (i < labels.length - 1) {
      drawArrow(x + w + 5, y + h / 2, x + w + gap - 8, y + h / 2, color(37, 99, 235));
    }
  }
}

// ------------------------------------------------------------
// BASELINE MODEL BRANCHES
// ------------------------------------------------------------

void drawBaselineBranches() {
  float startX = 255;
  float startY = 330;

  fill(15, 23, 42);
  textAlign(CENTER);
  textSize(24);
  text("Controlled Baseline Comparison", width / 2, 295);

  drawBlock(startX, startY, 210, 95, "MLP", "flattened event frames", color(255), color(99, 102, 241));
  drawBlock(startX + 260, startY, 210, 95, "CNN", "event-frame images", color(255), color(59, 130, 246));
  drawBlock(startX + 520, startY, 210, 95, "RNN / LSTM", "temporal sequences", color(255), color(16, 185, 129));
  drawBlock(startX + 780, startY, 210, 95, "SNN", "spike/event sequence", color(255), color(239, 68, 68));

  fill(71, 85, 105);
  textSize(16);
  text("Same dataset  •  same train/test split  •  same GPU  •  same evaluation metrics", width / 2, 460);
}

// ------------------------------------------------------------
// SNN MEMBRANE PLOT
// ------------------------------------------------------------

void drawSNNMembranePlot() {
  float x = 85;
  float y = 535;
  float w = 610;
  float h = 230;

  // Card
  stroke(203, 213, 225);
  strokeWeight(1.5);
  fill(255);
  rect(x, y, w, h, 18);

  fill(15, 23, 42);
  textAlign(LEFT);
  textSize(21);
  text("SNN neuron behavior: randomized membrane integration and spikes", x + 24, y + 36);

  // Plot region
  float px = x + 55;
  float py = y + 70;
  float pw = w - 90;
  float ph = h - 110;

  stroke(226, 232, 240);
  strokeWeight(1);
  for (int i = 0; i <= 5; i++) {
    float gy = py + i * ph / 5.0;
    line(px, gy, px + pw, gy);
  }

  // Threshold line
  float thY = py + ph * 0.28;
  stroke(239, 68, 68);
  strokeWeight(2);
  line(px, thY, px + pw, thY);

  fill(239, 68, 68);
  textSize(13);
  text("threshold", px + pw - 75, thY - 8);

  // Rest line
  float restY = py + ph * 0.82;
  stroke(148, 163, 184);
  strokeWeight(1);
  line(px, restY, px + pw, restY);

  fill(100, 116, 139);
  text("resting potential", px + pw - 120, restY + 18);

  // Randomized LIF-like trace
  noFill();
  stroke(37, 99, 235);
  strokeWeight(3);

  beginShape();

  float localMem = 0.18;
  float lastY = restY;

  for (int i = 0; i < 190; i++) {
    float input = 0.018 + 0.018 * noise(i * 0.11, frameCount * 0.01);
    float jitter = random(-0.002, 0.002);

    localMem = localMem * 0.985 + input + jitter;

    float vx = px + map(i, 0, 189, 0, pw);
    float vy = map(localMem, 0, 1.05, py + ph, py);

    if (localMem > threshold) {
      vertex(vx, vy);
      vertex(vx + 3, py + 8);
      vertex(vx + 8, restY);
      localMem = resetValue + random(0.02, 0.08);
    } else {
      vertex(vx, vy);
    }

    lastY = vy;
  }

  endShape();

  fill(37, 99, 235);
  noStroke();
  textSize(13);
  text("randomized input events accumulate voltage → spike → reset", px, y + h - 22);
}

// ------------------------------------------------------------
// METRICS DASHBOARD
// ------------------------------------------------------------

void drawMetricsDashboard() {
  float x = 765;
  float y = 535;
  float w = 650;
  float h = 230;

  stroke(203, 213, 225);
  strokeWeight(1.5);
  fill(255);
  rect(x, y, w, h, 18);

  fill(15, 23, 42);
  textAlign(LEFT);
  textSize(21);
  text("Evaluation dashboard", x + 24, y + 36);

  String[] metrics = {
    "Accuracy",
    "F1 score",
    "Latency",
    "Throughput",
    "GPU memory",
    "Energy",
    "Spike rate",
    "Firing rate"
  };

  for (int i = 0; i < metrics.length; i++) {
    float bx = x + 30 + (i % 4) * 150;
    float by = y + 70 + (i / 4) * 68;

    float value = 0.35 + 0.55 * noise(i * 0.7, frameCount * 0.015);

    stroke(226, 232, 240);
    fill(248, 250, 252);
    rect(bx, by, 125, 43, 10);

    noStroke();
    fill(71, 85, 105);
    textSize(13);
    text(metrics[i], bx + 10, by + 17);

    fill(14, 165, 233);
    rect(bx + 10, by + 27, 100 * value, 7, 4);
  }

  fill(100, 116, 139);
  textSize(13);
  text("General metrics apply to all models; spike metrics apply only to SNN.", x + 30, y + h - 23);
}

// ------------------------------------------------------------
// FOOTER
// ------------------------------------------------------------

void drawFooterMessage() {
  fill(30, 41, 59);
  textAlign(CENTER);
  textSize(17);
  text("Purpose: determine whether SNNs are competitive enough to justify deeper GPU simulation, scalability, and deployment analysis.", width / 2, 815);
}

// ------------------------------------------------------------
// DRAWING HELPERS
// ------------------------------------------------------------

void drawBlock(float x, float y, float w, float h, String title, String subtitle, color fillCol, color accentCol) {
  stroke(203, 213, 225);
  strokeWeight(1.4);
  fill(fillCol);
  rect(x, y, w, h, 16);

  noStroke();
  fill(accentCol);
  rect(x, y, 7, h, 16, 0, 0, 16);

  fill(15, 23, 42);
  textAlign(CENTER);
  textSize(20);
  text(title, x + w / 2, y + 33);

  fill(71, 85, 105);
  textSize(14);
  text(subtitle, x + w / 2, y + 60);
}

void drawArrow(float x1, float y1, float x2, float y2, color c) {
  stroke(c);
  strokeWeight(2.2);
  line(x1, y1, x2, y2);

  float angle = atan2(y2 - y1, x2 - x1);
  float arrowSize = 8;

  fill(c);
  noStroke();
  pushMatrix();
  translate(x2, y2);
  rotate(angle);
  triangle(0, 0, -arrowSize, -arrowSize / 2, -arrowSize, arrowSize / 2);
  popMatrix();
}

// ------------------------------------------------------------
// ANIMATED EVENT PARTICLES
// ------------------------------------------------------------

class FlowParticle {
  float progress;
  float speed;
  float yOffset;
  color c;

  FlowParticle() {
    reset();
    progress = random(1);
  }

  void reset() {
    progress = 0;
    speed = random(0.0015, 0.0045);
    yOffset = random(-22, 22);
    c = color(random(20, 80), random(120, 200), random(200, 255), 150);
  }

  void update() {
    progress += speed;
    if (progress > 1.0) reset();
  }

  void display() {
    float x1 = 70;
    float x2 = 1390;
    float y = 195 + yOffset + 7 * sin(progress * TWO_PI * 4);

    float x = lerp(x1, x2, progress);

    noStroke();
    fill(c);
    ellipse(x, y, 6, 6);
  }
}

// ------------------------------------------------------------
// BACKGROUND SPIKE DECORATION
// ------------------------------------------------------------

class Spike {
  float x;
  float y;
  float h;
  float phase;
  float speed;

  Spike() {
    x = random(width);
    y = random(100, height - 80);
    h = random(12, 32);
    phase = random(TWO_PI);
    speed = random(0.015, 0.045);
  }

  void update() {
    phase += speed;
  }

  void display() {
    float alpha = 35 + 45 * abs(sin(phase));
    stroke(14, 165, 233, alpha);
    strokeWeight(1.2);

    float spikeHeight = h * abs(sin(phase));
    line(x, y, x, y - spikeHeight);
  }
}