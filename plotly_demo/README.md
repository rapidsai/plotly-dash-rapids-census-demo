# Plotly-Dash + RAPIDS | Census 2020 Visualization

There are two versions for the same application, with all the views(described below) in both, single GPU and multi-GPU versions respectively.

Recommended GPU memory:

1. Single GPU version: 32GB+
2. Multi-GPU version: 2+ GPUs of 16GB+ each

```bash
# run and access single GPU version
cd plotly_demo
python app.py

# run and access multi GPU version
cd plotly_demo
python dask_app.py
```

## Snapshot Examples

### 1) Total Population View

![tp](https://user-images.githubusercontent.com/35873124/189298473-4a6895db-b5b3-49da-b47a-4a39233e7daf.png)

### 2) Migrating In View

![migin](https://user-images.githubusercontent.com/35873124/189298490-614a7efb-f172-4322-becc-eb79059bfbaa.png)

### 3) Stationary View

![stationary](https://user-images.githubusercontent.com/35873124/189298509-fb20b2af-3aee-4a12-9cba-885e3d2587f5.png)

### 4) Migrating Out View

![migout](https://user-images.githubusercontent.com/35873124/189298523-14983e47-38bd-4b73-97fa-6694b13f3362.png)

### 5) Net Migration View

![netmig](https://user-images.githubusercontent.com/35873124/189298570-64640492-4413-4d0e-a2be-aa2c91df6736.png)

#### Migration population to color mapping -

<b>Inward Migration</b>: Purple-Blue</br>
<b>Stationary</b>: Greens</br>
<b>Outward Migration</b>: Red Purples</br>

### 6) Population with Race view

![race](https://user-images.githubusercontent.com/35873124/189298602-11873dc3-89f2-4934-8208-b68e28e59d57.png)

#### Race to color mapping -

<b>White</b>: aqua</br>
<b>African American</b>: lime</br>
<b>American Indian</b>: yellow</br>
<b>Asian</b>: orange</br>
<b>Native Hawaiian</b>: blue</br>
<b>Other Race alone</b>: fuchsia</br>
<b>Two or More</b>: saddlebrown</br>
