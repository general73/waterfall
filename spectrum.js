/*
 * Copyright (c) 2019 Jeppe Ledet-Pedersen
 * This software is released under the MIT license.
 * See the LICENSE file for further details.
 */

'use strict';
//var colormaps = require('./colormap.js');

Spectrum.prototype.squeeze = function(value, out_min, out_max) {
    if (value <= this.min_db)
        return out_min;
    else if (value >= this.max_db)
        return out_max;
    else
        return Math.round((value - this.min_db) / (this.max_db - this.min_db) * out_max);
}

Spectrum.prototype.rowToImageData1 = function (bins) {
    for (var i = 0; i < this.imagedata.data.length; i += 4) {
        var cindex = this.squeeze(bins[i / 4], 0, 255);
        var color = this.colormap[cindex];
        this.imagedata.data[i + 0] = color[0];
        this.imagedata.data[i + 1] = color[1];
        this.imagedata.data[i + 2] = color[2];
        this.imagedata.data[i + 3] = 255;
    }
}

Spectrum.prototype.rowToImageData2 = function (bins) {
    for (var i = 0; i < this.imagedata.data.length; i += 4) {
        var ii = parseInt(i / 12) * 3;
        this.imagedata.data[i + 0] = this.squeeze(bins[ii + 0], 0, 255);
        this.imagedata.data[i + 1] = this.squeeze(bins[ii + 1], 0, 255);
        this.imagedata.data[i + 2] = this.squeeze(bins[ii + 2], 0, 255);
        this.imagedata.data[i + 3] = 255;
    }
}

Spectrum.prototype.rowToImageData = Spectrum.prototype.rowToImageData2;

Spectrum.prototype.addWaterfallRow = function (bins) {
    // Shift waterfall 1 row down
    this.ctx_wf.drawImage(this.ctx_wf.canvas,
        0, 0, this.wf_size, this.wf_rows - 1,
        0, 1, this.wf_size, this.wf_rows - 1);

    // Draw new line on waterfall canvas
    this.rowToImageData(bins);
    this.ctx_wf.putImageData(this.imagedata, 0, 0);

    var width = this.ctx.canvas.width;
    var height = this.ctx.canvas.height;

    // Copy scaled FFT canvas to screen. Only copy the number of rows that will
    // fit in waterfall area to avoid vertical scaling.
    this.ctx.imageSmoothingEnabled = false;
    var rows = Math.min(this.wf_rows, height - this.spectrumHeight);
    this.ctx.drawImage(this.ctx_wf.canvas,
        0, 0, this.wf_size, rows,
        0, this.spectrumHeight, width, height - this.spectrumHeight);
}

Spectrum.prototype.drawFFT = function (bins) {
    this.ctx.beginPath();
    this.ctx.moveTo(-1, this.spectrumHeight + 1);
    for (var i = 0; i < bins.length; i++) {
        var y = this.spectrumHeight - this.squeeze(bins[i], 0, this.spectrumHeight);
        if (y > this.spectrumHeight - 1)
            y = this.spectrumHeight + 1; // Hide underflow
        if (y < 0)
            y = 0;
        if (i == 0)
            this.ctx.lineTo(-1, y);
        this.ctx.lineTo(i, y);
        if (i == bins.length - 1)
            this.ctx.lineTo(this.wf_size + 1, y);
    }
    this.ctx.lineTo(this.wf_size + 1, this.spectrumHeight + 1);
    this.ctx.strokeStyle = "#fefefe";
    this.ctx.stroke();
}

Spectrum.prototype.drawSpectrum = function (bins) {
    var width = this.ctx.canvas.width;
    var height = this.ctx.canvas.height;

    // Fill with black
    this.ctx.fillStyle = "black";
    this.ctx.fillRect(0, 0, width, height);

    // FFT averaging
    if (this.averaging > 0) {
        if (!this.binsAverage || this.binsAverage.length != bins.length) {
            this.binsAverage = Array.from(bins);
        } else {
            for (var i = 0; i < bins.length; i++) {
                this.binsAverage[i] += this.alpha * (bins[i] - this.binsAverage[i]);
            }
        }
        bins = this.binsAverage;
    }

    // Max hold
    if (this.maxHold) {
        if (!this.binsMax || this.binsMax.length != bins.length) {
            this.binsMax = Array.from(bins);
        } else {
            for (var i = 0; i < bins.length; i++) {
                if (bins[i] > this.binsMax[i]) {
                    this.binsMax[i] = bins[i];
                } else {
                    // Decay
                    this.binsMax[i] = 1.0025 * this.binsMax[i];
                }
            }
        }
    }

    // Do not draw anything if spectrum is not visible
    if (this.ctx_axes.canvas.height < 1) {
        return;
    }

    // Scale for FFT
    this.ctx.save();
    this.ctx.scale(width / this.wf_size, 1);

    // Draw maxhold
    if (this.maxHold) {
        this.drawFFT(this.binsMax);
    }

    // Draw FFT bins
    this.drawFFT(bins);

    // Restore scale
    this.ctx.restore();

    // Fill scaled path
    this.ctx.fillStyle = this.gradient;
    this.ctx.fill();

    // Copy axes from offscreen canvas
    this.ctx.drawImage(this.ctx_axes.canvas, 0, 0);
}

Spectrum.prototype.updateAxes = function () {
    var width = this.ctx_axes.canvas.width;
    var height = this.ctx_axes.canvas.height;

    // Clear axes canvas
    this.ctx_axes.clearRect(0, 0, width, height);

    // Draw axes
    this.ctx_axes.font = "12px sans-serif";
    this.ctx_axes.fillStyle = "white";
    this.ctx_axes.textBaseline = "middle";

    this.ctx_axes.textAlign = "left";
    var step = 10;
    for (var i = this.min_db + 10; i <= this.max_db - 10; i += step) {
        var y = height - this.squeeze(i, 0, height);
        this.ctx_axes.fillText(i, 5, y);

        this.ctx_axes.beginPath();
        this.ctx_axes.moveTo(20, y);
        this.ctx_axes.lineTo(width, y);
        this.ctx_axes.strokeStyle = "rgba(200, 200, 200, 0.10)";
        this.ctx_axes.stroke();
    }

    this.ctx_axes.textBaseline = "bottom";
    for (var i = 0; i < this.ticksHz; i++) {
        var x = Math.round(width / (this.ticksHz - 1)) * i;

        if (this.spanHz > 0) {
            var adjust = 0;
            if (i === 0) {
                this.ctx_axes.textAlign = "left";
                adjust = 3;
            } else if (i === (this.ticksHz - 1)) {
                this.ctx_axes.textAlign = "right";
                adjust = -3;
            } else {
                this.ctx_axes.textAlign = "center";
            }

            var freqFactor = (2 / (this.ticksHz - 1)) * i - 1;  // range <-1; +1>
            var freq = this.centerHz + this.spanHz * freqFactor / 2;
            if (this.centerHz + this.spanHz > 1e6) {
                freq = freq / 1e6;
                freq = (freq * 100).toFixed(0) / 100;
                freq += "M";
            }
            else if (this.centerHz + this.spanHz > 1e3) {
                freq = freq / 1e3;
                freq = (freq * 100).toFixed(0) / 100;
                freq += "k";
            }

            if (this.horizontalAxisPosition === 'both') {
                this.ctx_axes.fillText(freq, x + adjust, height);
                this.ctx_axes.fillText(freq, x + adjust, 12);
            } else if (this.horizontalAxisPosition === 'bottom') {
                this.ctx_axes.fillText(freq, x + adjust, height);
            } else if (this.horizontalAxisPosition === 'top') {
                this.ctx_axes.fillText(freq, x + adjust, 12);
            }
        }

        this.ctx_axes.beginPath();
        this.ctx_axes.moveTo(x, 0);
        this.ctx_axes.lineTo(x, height);
        this.ctx_axes.strokeStyle = "rgba(200, 200, 200, 0.10)";
        this.ctx_axes.stroke();
    }
}

Spectrum.prototype.horizontalZoom = function (zoomNumber, centerFreq) {
    //940000000 40000000
    zoomNumber = zoomNumber == undefined ? 0 : zoomNumber;
    if (isNaN(zoomNumber) || zoomNumber >= 100) {
        return;
    }
    //if (zoomNumber <= 1) {
    //    this.zoomNumber = zoomNumber;
    //    this.setCenterHz(this.originalCenterHz);
    //    this.setSpanHz(this.originalSpanHz);
    //    this.filterData = function (data) {
    //        return data;
    //    }
    //    return;
    //}
    this.zoomNumber = zoomNumber;
    centerFreq = centerFreq ? centerFreq : this.centerHz;

    this.setCenterHz(this.originalCenterHz, centerFreq);
    this.setSpanHz(this.originalSpanHz, this.originalSpanHz - (this.originalSpanHz * zoomNumber / 100));
    //this.setSpanHz(40000000 / zoomNumber);

    this.filterData = function (data) {
        //data = data.slice(data.length / zoomNumber / 2, -(data.length / zoomNumber / 2));
        let step = this.originalSpanHz / data.length;
        let margin = this.centerHz - this.originalCenterHz;
        let offset = margin / step;

        let leftIndex = data.length * zoomNumber / 100 / 2;
        let rightIndex = data.length - (data.length * zoomNumber / 100 / 2);
        leftIndex += offset;
        rightIndex += offset;
        //this.wf_size = rightIndex - leftIndex;
        if (leftIndex > data.length || rightIndex < 0) {
            let emptyData = [];
            emptyData.length = rightIndex - leftIndex;
            emptyData.fill(this.min_db);
            return emptyData;
        }

        let rightAppendCount = 0;
        if (rightIndex > data.length) {
            rightAppendCount = rightIndex - data.length;
            rightAppendCount = Math.ceil(rightAppendCount);
            rightAppendCount = rightAppendCount > data.length ? data.length : rightAppendCount;
            rightIndex = data.length;
        }

        let leftPrependCount = 0;
        if (leftIndex < 0) {
            leftPrependCount = leftIndex * (-1);
            leftPrependCount = Math.floor(leftPrependCount);
            leftPrependCount = leftPrependCount > data.length ? data.length : leftPrependCount;
            leftIndex = 0;
        }

        data = data.slice(leftIndex, rightIndex);

        if (rightAppendCount > 0) {
            let len = data.length;
            data.length = len + rightAppendCount;
            data.fill(this.min_db, len, data.length);
        }

        if (leftPrependCount > 0) {
            let prependData = [];
            prependData.length = leftPrependCount;
            prependData.fill(this.min_db);
            data = prependData.concat(data)
        }

        return data;
    }
}

Spectrum.prototype.filterData = function (data) {
    //return data.slice(50, -550)
    return data;
}

Spectrum.prototype.addData = function (data) {
    if (!this.paused) {

        data = this.filterData(data);

        if (data.length != this.wf_size) {
            this.wf_size = data.length;
            this.ctx_wf.canvas.width = data.length;
            this.ctx_wf.fillStyle = "black";
            this.ctx_wf.fillRect(0, 0, this.wf.width, this.wf.height);
            this.imagedata = this.ctx_wf.createImageData(data.length, 1);
        }

        this.drawSpectrum(data);
        this.addWaterfallRow(data);
        this.resize();
    }
}

Spectrum.prototype.updateSpectrumRatio = function () {
    this.spectrumHeight = Math.round(this.canvas.height * this.spectrumPercent / 100.0);

    this.gradient = this.ctx.createLinearGradient(0, 0, 0, this.spectrumHeight);
    for (var i = 0; i < this.colormap.length; i++) {
        var c = this.colormap[this.colormap.length - 1 - i];
        this.gradient.addColorStop(i / this.colormap.length,
            "rgba(" + c[0] + "," + c[1] + "," + c[2] + ", 1.0)");
    }
}

Spectrum.prototype.resize = function () {
    var width = this.canvas.clientWidth;
    var height = this.canvas.clientHeight;

    if (this.canvas.width != width ||
        this.canvas.height != height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.updateSpectrumRatio();
    }

    if (this.axes.width != width ||
        this.axes.height != this.spectrumHeight) {
        this.axes.width = width;
        this.axes.height = this.spectrumHeight;
        this.updateAxes();
    }

}

Spectrum.prototype.setSpectrumPercent = function (percent) {
    if (percent >= 0 && percent <= 100) {
        this.spectrumPercent = percent;
        this.updateSpectrumRatio();
    }
}

Spectrum.prototype.incrementSpectrumPercent = function () {
    if (this.spectrumPercent + this.spectrumPercentStep <= 100) {
        this.setSpectrumPercent(this.spectrumPercent + this.spectrumPercentStep);
    }
}

Spectrum.prototype.decrementSpectrumPercent = function () {
    if (this.spectrumPercent - this.spectrumPercentStep >= 0) {
        this.setSpectrumPercent(this.spectrumPercent - this.spectrumPercentStep);
    }
}

Spectrum.prototype.toggleColor = function () {
    this.colorindex++;
    if (this.colorindex >= colormaps.length)
        this.colorindex = 0;
    this.colormap = colormaps[this.colorindex];
    this.updateSpectrumRatio();
}

Spectrum.prototype.toggleColorType = function () {
    this.wt_useColorMap = !this.wt_useColorMap;
    if (this.wt_useColorMap)
        this.rowToImageData = this.rowToImageData2;
    else
        this.rowToImageData = this.rowToImageData1;
}

Spectrum.prototype.setRange = function (min_db, max_db, temporary) {
    this.min_db = min_db;
    this.max_db = max_db;
    if (!temporary) {
        this.max_db_original = this.max_db;
        this.min_db_original = this.min_db;
    }
    this.updateAxes();
}

Spectrum.prototype.rangeUp = function () {
    this.setRange(this.min_db - 5, this.max_db - 5, true);
}

Spectrum.prototype.rangeDown = function () {
    this.setRange(this.min_db + 5, this.max_db + 5, true);
}

Spectrum.prototype.rangeIncrease = function () {
    this.setRange(this.min_db - 5, this.max_db + 5, true);
}

Spectrum.prototype.rangeDecrease = function () {
    if (this.max_db - this.min_db > 10)
        this.setRange(this.min_db + 5, this.max_db - 5, true);
}

Spectrum.prototype.setCenterHz = function (hz, zoomHz) {
    this.centerHz = zoomHz ? zoomHz : hz;
    this.originalCenterHz = hz;
    this.updateAxes();
}

Spectrum.prototype.setSpanHz = function (hz, zoomHz) {
    this.spanHz = zoomHz ? zoomHz : hz;
    this.originalSpanHz = hz;
    this.updateAxes();
}

Spectrum.prototype.setAveraging = function (num) {
    if (num >= 0) {
        this.averaging = num;
        this.alpha = 2 / (this.averaging + 1)
    }
}

Spectrum.prototype.incrementAveraging = function () {
    this.setAveraging(this.averaging + 1);
}

Spectrum.prototype.decrementAveraging = function () {
    if (this.averaging > 0) {
        this.setAveraging(this.averaging - 1);
    }
}

Spectrum.prototype.setPaused = function (paused) {
    this.paused = paused;
}

Spectrum.prototype.togglePaused = function () {
    this.setPaused(!this.paused);
}

Spectrum.prototype.setMaxHold = function (maxhold) {
    this.maxHold = maxhold;
    this.binsMax = undefined;
}

Spectrum.prototype.toggleMaxHold = function () {
    this.setMaxHold(!this.maxHold);
}

Spectrum.prototype.toggleFullscreen = function () {
    if (!this.fullscreen) {
        if (this.canvas.requestFullscreen) {
            this.canvas.requestFullscreen();
        } else if (this.canvas.mozRequestFullScreen) {
            this.canvas.mozRequestFullScreen();
        } else if (this.canvas.webkitRequestFullscreen) {
            this.canvas.webkitRequestFullscreen();
        } else if (this.canvas.msRequestFullscreen) {
            this.canvas.msRequestFullscreen();
        }
        this.fullscreen = true;
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        this.fullscreen = false;
    }
}

Spectrum.prototype.onKeypress = function (e) {
    if (e.key == " ") {
        this.togglePaused();
    } else if (e.key == "f") {
        this.toggleFullscreen();
    } else if (e.key == "c") {
        this.toggleColor();
    } else if (e.key == "v") {
        this.toggleColorType();
    } else if (e.key == "ArrowUp") {
        this.rangeUp();
    } else if (e.key == "ArrowDown") {
        this.rangeDown();
    } else if (e.key == "ArrowLeft") {
        this.horizontalZoom(this.zoomNumber, this.centerHz + this.spanHz / 10)
    } else if (e.key == "ArrowRight") {
        this.horizontalZoom(this.zoomNumber, this.centerHz - this.spanHz / 10)
    } else if (e.key == "5") {
        this.horizontalZoom(0, this.originalCenterHz)
    } else if (e.key == "6") {
        this.horizontalZoom(this.zoomNumber + 10)
    } else if (e.key == "4") {
        this.horizontalZoom(this.zoomNumber - 10)
    } else if (e.key == "8") {
        this.rangeDecrease();
    } else if (e.key == "2") {
        this.rangeIncrease();
    } else if (e.key == "0") {
        this.setRange(this.min_db_original, this.max_db_original);
    } else if (e.key == "s") {
        this.incrementSpectrumPercent();
    } else if (e.key == "w") {
        this.decrementSpectrumPercent();
    } else if (e.key == "+") {
        this.incrementAveraging();
    } else if (e.key == "-") {
        this.decrementAveraging();
    } else if (e.key == "m") {
        this.toggleMaxHold();
    }
}

function Spectrum(id, options) {
    // Handle options
    this.centerHz = (options && options.centerHz) ? options.centerHz : 0;
    this.spanHz = (options && options.spanHz) ? options.spanHz : 0;
    this.wf_size = (options && options.wf_size) ? options.wf_size : 0;
    this.wf_rows = (options && options.wf_rows) ? options.wf_rows : 2048;
    this.spectrumPercent = (options && options.spectrumPercent) ? options.spectrumPercent : 25;
    this.spectrumPercentStep = (options && options.spectrumPercentStep) ? options.spectrumPercentStep : 5;
    this.averaging = (options && options.averaging) ? options.averaging : 0.5;
    this.ticksHz = (options && options.ticksHz) ? options.ticksHz : 11;
    this.horizontalAxisPosition = (options && options.horizontalAxisPosition) ? options.horizontalAxisPosition : 'bottom';  // either 'top', 'bottom' or 'both'
    this.maxHold = (options && options.maxHold) ? options.maxHold : false;
    this.zoomNumber = 0;

    // Setup state
    this.paused = false;
    this.fullscreen = false;
    //this.min_db = -120;
    //this.max_db = -20;
    this.min_db = 0;
    this.max_db = 255;
    this.max_db_original = this.max_db;
    this.min_db_original = this.min_db;
    this.spectrumHeight = 0;

    // Colors
    this.colorindex = 0;
    this.colormap = colormaps[0];
    this.wt_useColorMap = false;

    // Create main canvas and adjust dimensions to match actual
    this.canvas = document.getElementById(id);
    if (this.canvas === null) {
        throw "There is no <canvas> declared with id #" + id
    }
    this.canvas.height = this.canvas.clientHeight;
    this.canvas.width = this.canvas.clientWidth;
    this.ctx = this.canvas.getContext("2d");
    this.ctx.fillStyle = "black";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Create offscreen canvas for axes
    this.axes = document.createElement("canvas");
    this.axes.height = 1; // Updated later
    this.axes.width = this.canvas.width;
    this.ctx_axes = this.axes.getContext("2d");

    // Create offscreen canvas for waterfall
    this.wf = document.createElement("canvas");
    this.wf.height = this.wf_rows;
    this.wf.width = this.wf_size;
    this.ctx_wf = this.wf.getContext("2d");

    // Trigger first render
    this.setAveraging(this.averaging);
    this.updateSpectrumRatio();
    this.resize();
}

//module.exports = Spectrum;
