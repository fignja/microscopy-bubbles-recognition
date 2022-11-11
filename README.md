# microscopy-bubbles-recognition
Microscopy bubbles segmentation and size measuring system for scientific purposes by Korolev I.A.

This project is designed to automatically label air bubbles on microscopic images.
Any cuda videocard is required. Minimal requerement is GTX650 for segmentation, and 1650GTX 4G VRAm for models training.

Just put imgs in segmentator and run .<pre>pig.py</pre> to process the image.
Then run the editor .<pre>edit.py</pre> to get the result and fix some segmentation issues if required.
![Autosegmentation](./examples/process.png)
