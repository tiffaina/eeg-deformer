# eeg-deformer

Paper: https://ieeexplore.ieee.org/document/10763464	

Paper's Github: https://github.com/yi-ding-cs/EEG-Deformer	

##Datasets 
 DREAMER: Download from [here](https://zenodo.org/records/546113?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc2NTI5NzY5OSwiZXhwIjoxNzY3ODMwMzk5fQ.eyJpZCI6IjQ1NzgyYWYwLTYyMWYtNDdmNi1iMjNkLTYzNDYyOTgzYzA0NCIsImRhdGEiOnt9LCJyYW5kb20iOiJlMDNmZjgxOTUwZmVhOWIyNzAyNDY3OTFiNmNmYzEzMiJ9.b_QSFTtWWHXbwtEl91rdt5m5O77UQZYJUDOriUHi8-H-2Ya3zBvRSEcrsshyX1dZKLHGEKf1DoqSIqJKUzAfxQ)
Fatigue: Download from [here](https://drive.google.com/file/d/1KwPPSHN14MAbhszGqC1O5nRei7oqllxl/view?usp=sharing)


### Path Adjustment
In the first cell of the  notebook, you will find the `PATH` variables. Update these to point to your local directory or personal Google Drive:

```python
# Change these paths to your local storage or Drive location
FATIGUE_DATA_PATH = '/content/drive/MyDrive/path_to/data_eeg_FATIG_FTG'
DREAMER_MAT_PATH = '/content/drive/MyDrive/path_to/DREAMER.mat'
