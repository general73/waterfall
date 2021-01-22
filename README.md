# HTML Canvas/WebSockets Waterfall

This is a small experiment to create a waterfall plot with HTML Canvas and WebSockets to stream live FFT data from an SDR:

![alt](img/waterfall.png)

## Installation

```bash
$ npm install git+https://github.com/jledet/waterfall.git
```

## Usage

```ecmascript 6
import Spectrum from 'fft-waterfall';
```

## Files definition
`spectrum.js` contains the main JavaScript source code for the plot, while `colormap.js` contains colormaps generated using ``make_colormap.py``.

`index.html`, `style.css`, `script.js` contain an example page that receives FFT data on a WebSocket and plots it on the waterfall plot.

`server.py` contains a example [Bottle](https://bottlepy.org/docs/dev/) and [gevent-websocket](https://pypi.org/project/gevent-websocket/) server 
that broadcasts FFT data to connected clients. The FFT data is generated using [GNU radio](https://www.gnuradio.org/) using a USRP but it 
should be fairly easy to change it to a different SDR.
