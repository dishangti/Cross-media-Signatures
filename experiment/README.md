## Experimental Group
### Folder Structure
- Folder "rect" contains cropped signatures from original data.
- Folder "square" contains squared signatures from folder "rect".
- Folder "bin" contains binarized signatures from folder "square".
- Folder "skel" contains skeletonized signatures from folder "bin".

Volunteers numbered `34, 36` are written with left hand.

Notice that data in folder "square" of number `51, 68, 79, 93, 94, 95, 95, 100` has been abandoned and then renumbered in folder "bin" and "skel".

### File Description
- File "Bezier.py" is for Bezier model feature extraction.
- File "Bezier.ipynb" is for Bezier model figure generation.
- File "Classic.ipynb" is for classic model feature extraction.
- File "cnn.py" is classifier.
- File "snn2.py", "snn3.py" and "snn4.py" are comparators for iPad, phone (in hand), phone (on table) respectively.
- File "pvalue.ipynb" is for p-value calculation and then heatmap figure generation.
- File "nn_graph.ipynb" is for neural network figures generation.