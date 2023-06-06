## Control Group
### Folder Structure
- Folder "rect" contains cropped signatures from original data.
- Folder "square" contains squared signatures from folder "rect".
- Folder "bin" contains binarized signatures from folder "square".
- Folder "skel" contains skeletonized signatures from folder "bin".

### File Description
- File "Bezier.py" is for Bezier model feature extraction.
- File "Classic.ipynb" is for classic model feature extraction.
- File "cnn.py" and "snn2.py" are classifier and comparator respectively.
- File "pvalue.ipynb" is for p-value calculation and then for heatmap figure generation.