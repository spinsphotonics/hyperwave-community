/**
 * Interactive GPU cost/time calculator for Hyperwave Community docs.
 * Users input device dimensions (um) and resolution (nm),
 * and the calculator shows estimated time, cost, and OOM warnings per GPU.
 */
document.addEventListener("DOMContentLoaded", function () {
  var form = document.getElementById("gpu-calc-form");
  if (!form) return;

  var GPUS = [
    { name: "B200", vram: 192, speed: 25, multiplier: 2.5, maxVoxelsM: 700 },
    { name: "H200", vram: 141, speed: 16, multiplier: 2.0, maxVoxelsM: 510 },
    { name: "H100", vram: 80, speed: 13, multiplier: 1.5, maxVoxelsM: 290 },
    { name: "A100-80GB", vram: 80, speed: 9, multiplier: 1.0, maxVoxelsM: 290 },
    { name: "A100-40GB", vram: 40, speed: 7, multiplier: 0.8, maxVoxelsM: 145 },
    { name: "L40S", vram: 48, speed: 5, multiplier: 0.7, maxVoxelsM: 175 },
    { name: "A10G", vram: 24, speed: 3, multiplier: 0.4, maxVoxelsM: 85 },
    { name: "T4", vram: 16, speed: 2, multiplier: 0.3, maxVoxelsM: 58 },
  ];

  var BASE_RATE = 0.000278; // credits per second (at 1.0x multiplier)
  var USD_PER_CREDIT = 10;

  form.addEventListener("input", calculate);
  calculate();

  function calculate() {
    var xUm = parseFloat(document.getElementById("calc-x").value) || 0;
    var yUm = parseFloat(document.getElementById("calc-y").value) || 0;
    var zUm = parseFloat(document.getElementById("calc-z").value) || 0;
    var resNm = parseFloat(document.getElementById("calc-res").value) || 20;
    var maxSteps =
      parseInt(document.getElementById("calc-steps").value, 10) || 20000;

    if (xUm <= 0 || yUm <= 0 || zUm <= 0 || resNm <= 0) {
      document.getElementById("gpu-calc-results").innerHTML =
        "<p>Enter valid dimensions and resolution above.</p>";
      return;
    }

    var resUm = resNm / 1000;
    var xCells = Math.ceil(xUm / resUm);
    var yCells = Math.ceil(yUm / resUm);
    var zCells = Math.ceil(zUm / resUm);
    var gridPoints = xCells * yCells * zCells;
    var voxelsM = gridPoints / 1e6;

    var infoHtml =
      "<p><strong>Grid:</strong> " +
      xCells + " x " + yCells + " x " + zCells +
      " = " + voxelsM.toFixed(1) + "M voxels</p>";

    var html =
      infoHtml +
      '<table class="table sortable">' +
      "<thead><tr>" +
      "<th>GPU</th>" +
      "<th>VRAM (GB)</th>" +
      "<th>Est. Time</th>" +
      "<th>Est. Credits</th>" +
      "<th>Est. Cost (USD)</th>" +
      "<th>Status</th>" +
      "</tr></thead><tbody>";

    for (var i = 0; i < GPUS.length; i++) {
      var gpu = GPUS[i];
      var oom = voxelsM > gpu.maxVoxelsM;
      var rowClass = oom ? ' class="gpu-oom-row"' : "";

      if (oom) {
        html +=
          "<tr" + rowClass + ">" +
          "<td><code>" + gpu.name + "</code></td>" +
          "<td>" + gpu.vram + "</td>" +
          "<td>--</td>" +
          "<td>--</td>" +
          "<td>--</td>" +
          '<td class="gpu-oom-badge">OOM (' +
          gpu.maxVoxelsM + "M max)</td>" +
          "</tr>";
      } else {
        var simSeconds = (gridPoints * maxSteps) / (gpu.speed * 1e9);
        var credits = simSeconds * BASE_RATE * gpu.multiplier;
        var costUsd = credits * USD_PER_CREDIT;

        var timeStr;
        if (simSeconds < 60) {
          timeStr = simSeconds.toFixed(0) + "s";
        } else if (simSeconds < 3600) {
          timeStr =
            Math.floor(simSeconds / 60) +
            "m " +
            Math.round(simSeconds % 60) +
            "s";
        } else {
          timeStr =
            Math.floor(simSeconds / 3600) +
            "h " +
            Math.round((simSeconds % 3600) / 60) +
            "m";
        }

        html +=
          "<tr" + rowClass + ">" +
          "<td><code>" + gpu.name + "</code></td>" +
          "<td>" + gpu.vram + "</td>" +
          "<td>" + timeStr + "</td>" +
          "<td>" + credits.toFixed(4) + "</td>" +
          "<td>$" + costUsd.toFixed(2) + "</td>" +
          '<td class="gpu-ok-badge">OK</td>' +
          "</tr>";
      }
    }

    html += "</tbody></table>";
    html +=
      '<p class="gpu-calc-note">Estimates assume ' +
      maxSteps.toLocaleString() +
      " max steps. Actual cost may be lower if the simulation converges early.</p>";

    document.getElementById("gpu-calc-results").innerHTML = html;
  }
});
